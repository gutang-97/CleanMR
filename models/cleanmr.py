# coding: utf-8

import os
from pickle import FALSE
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, degree
import torch_geometric
from tqdm import tqdm
from common.abstract_recommender import GeneralRecommender
from common.loss import BPRLoss, EmbLoss
from common.init import xavier_uniform_initialization
from torch_scatter import scatter_mean, scatter_sum, scatter_softmax
import random

class CleanMR(GeneralRecommender):
    def __init__(self, config, dataset):
        super(CleanMR, self).__init__(config, dataset)

        num_user = self.n_users
        num_item = self.n_items
        batch_size = config['train_batch_size']         
        dim_x = config['embedding_size']                
        self.feat_embed_dim = config['feat_embed_dim']  
        self.n_layers = config['n_mm_layers']           
        self.knn_k = config['knn_k']                     
        self.mm_image_weight = config['mm_image_weight'] 
        has_id = True
        self.dropout = nn.Dropout(p=0.3)
        self.batch_size = batch_size
        self.num_user = num_user                       
        self.num_item = num_item                       
        self.k = 40
        self.num_interest = 3
        self.aggr_mode = config['aggr_mode']           
        self.user_aggr_mode = 'softmax'
        self.num_layer = 1
        self.dataset = dataset
        #self.construction = 'weighted_max'
        self.construction = 'cat'
        self.reg_weight = config['reg_weight']        
        self.drop_rate = 0.1
        self.v_rep = None
        self.t_rep = None

        self.review_score = None
        self.review_soft_list = None
        
        self.dim_latent = 384
        self.mm_adj = None
        self.a_feat = None
        
        dataset_path = os.path.abspath(config['data_path'] + config['dataset'])

        mm_adj_file = os.path.join(dataset_path, 'mm_adj_{}.pt'.format(self.knn_k))

        if self.v_feat is not None:
            self.v_feat = nn.Embedding.from_pretrained(self.v_feat, freeze=False).weight  

        if self.t_feat is not None:
            self.t_feat = nn.Embedding.from_pretrained(self.t_feat, freeze=False).weight

        
        if os.path.exists(mm_adj_file):
            self.mm_adj = torch.load(mm_adj_file)                                           
            # indices, image_adj = self.get_knn_adj_mat(self.v_feat.detach())
        else:
            if self.v_feat is not None:
                indices, image_adj = self.get_knn_adj_mat(self.image_embedding.weight.detach())
                self.mm_adj = image_adj
            if self.t_feat is not None:
                indices, text_adj = self.get_knn_adj_mat(self.text_embedding.weight.detach())
                self.mm_adj = text_adj
            if self.v_feat is not None and self.t_feat is not None:
                self.mm_adj = self.mm_image_weight * image_adj + (1.0 - self.mm_image_weight) * text_adj
                del text_adj
                del image_adj
            torch.save(self.mm_adj, mm_adj_file)
        
        train_interactions = dataset.inter_matrix(form='coo').astype(np.float32)                     
        # train_interactions4user = self.add_noise(train_interactions)
        self.ui_graph = self.matrix_to_tensor(self.csr_norm(train_interactions, mean_flag=False))
        self.iu_graph = self.matrix_to_tensor(self.csr_norm(train_interactions.T, mean_flag=False))

        self.extend_ui_graph_t = self.get_extend_ui_mat(self.t_feat,self.ui_graph)
        self.extend_ui_graph_v = self.get_extend_ui_mat(self.v_feat,self.ui_graph)
        # self.extend_ui_graph = self.extend_ui_graph_t * 0.8 + self.extend_ui_graph_v * 0.2
        self.extend_ui_graph = self.extend_ui_graph_t
        

        dense_interactions = torch.tensor(train_interactions.toarray(),dtype=torch.float32).to(self.device)
        self.dense_adj_mask = (dense_interactions != 0).float().transpose(1,0)
        self.dense_adj_bool = (self.dense_adj_mask.sum(-1) == 0.).float()

        self.user_id_embedding = nn.Parameter(nn.init.uniform_(torch.zeros(self.n_users, self.dim_latent)))

        self.MLP_t = nn.Linear(self.t_feat.shape[1], self.dim_latent)
        nn.init.uniform_(self.MLP_t.weight,a=-1.0,b=1.0)
        self.MLP_v = nn.Linear(self.v_feat.shape[1], self.dim_latent)
        nn.init.uniform_(self.MLP_v.weight,a=-1.0,b=1.0)

        self.MLP_t_1 = nn.Linear(self.dim_latent, self.dim_latent)
        self.MLP_v_1 = nn.Linear(self.dim_latent, self.dim_latent)

        
        self.text_decompose = ModalDecompose(self.dim_latent,self.device,num_user,num_item)
        self.text_club = CLUB(self.dim_latent)
        self.image_decompose = ModalDecompose(self.dim_latent,self.device,num_user,num_item)
        self.image_club = CLUB(self.dim_latent)
        
        self.cross_rec = Corss_Rec(self.dim_latent,self.dim_latent,self.dim_latent)

        self.user_linear = nn.Linear(self.dim_latent,self.dim_latent)
        
        self.t_common, self.v_common, self.t_specific, self.v_specific = None, None, None, None

        self.preference = nn.Parameter(nn.init.xavier_normal_(torch.tensor(
            np.random.randn(num_user, self.dim_latent), dtype=torch.float32, requires_grad=True),
            gain=1).to(self.device))  
        self.IB_model = IBModel(self.dim_latent,self.preference,num_user,self.device)
        self.FreeIB_model = FreeIBModel(self.dim_latent,self.preference,num_user,self.device)


        self.item_id = nn.Parameter(nn.init.xavier_normal_(torch.tensor(
            np.random.randn(num_item, self.dim_latent), dtype=torch.float32, requires_grad=True),
            gain=1).to(self.device)) 


        if self.a_feat:
            self.a_gcn = GCN(self.dataset, batch_size, num_user, num_item, dim_x, self.aggr_mode,        
                            num_layer=self.num_layer, has_id=has_id, dropout=self.drop_rate, dim_latent=self.dim_latent,
                            device=self.device, features=self.t_feat)
            self.a_preference = nn.Parameter(nn.init.xavier_normal_(torch.tensor(
            np.random.randn(num_user, self.dim_latent), dtype=torch.float32, requires_grad=True),
            gain=1).to(self.device))  
            
            self.MLP_a = nn.Linear(self.a_feat.shape[1], self.dim_latent)
        self.t_score = None 
        self.v_score = None
         
        self.u_embed = None
        self.i_embed = None
   
    def add_noise(self,interactions,rate=0.8):
        num_non_zero = len(interactions.data) 
        num_noise_edges = int(num_non_zero * rate)

        num_rows, num_cols = interactions.shape

        non_zero_row_indices = interactions.row
        non_zero_col_indices = interactions.col

        zeros_mask = np.ones((num_rows, num_cols), dtype=bool)

        zeros_mask[non_zero_row_indices, non_zero_col_indices] = False

        zero_positions = np.column_stack(np.where(zeros_mask))

        random_zero_indices = random.sample(range(len(zero_positions)), num_noise_edges)

        noisy_train_interactions = interactions.copy().tocsr()

        rows_to_modify = zero_positions[random_zero_indices, 0]
        cols_to_modify = zero_positions[random_zero_indices, 1]
        
        noisy_train_interactions[rows_to_modify, cols_to_modify] = 1
        noisy_train_interactions = noisy_train_interactions.tocoo()
        noisy_train_interactions.data = np.ones_like(noisy_train_interactions.data)
        return noisy_train_interactions
    
    def pca(self, x, k=2):
        x_mean = torch.mean(x, 0)
        x = x - x_mean
        cov_matrix = torch.matmul(x.t(), x) / (x.size(0) - 1)
        eigenvalues, eigenvectors = torch.linalg.eig(cov_matrix)
        sorted_eigenvalues, indices = torch.sort(eigenvalues.real, descending=True)
        components = eigenvectors[:, indices[:k]]
        x_pca = torch.matmul(x, components)
        return x_pca
    
    def matrix_to_tensor(self, cur_matrix):
        if type(cur_matrix) != sp.coo_matrix:
            cur_matrix = cur_matrix.tocoo()  #
        ui_indices = torch.from_numpy(np.vstack((cur_matrix.row, cur_matrix.col)).astype(np.int64))  
        iu_indices = torch.from_numpy(np.vstack((cur_matrix.col, cur_matrix.row)).astype(np.int64))
        values = torch.from_numpy(cur_matrix.data)  #
        shape = torch.Size(cur_matrix.shape)

        return torch.sparse.FloatTensor(ui_indices, values, shape).to(torch.float32).cuda()

    def csr_norm(self, csr_mat, mean_flag=False):
        rowsum = np.array(csr_mat.sum(1))
        rowsum = np.power(rowsum+1e-8, -0.5).flatten()
        rowsum[np.isinf(rowsum)] = 0.
        rowsum_diag = sp.diags(rowsum)

        colsum = np.array(csr_mat.sum(0))
        colsum = np.power(colsum+1e-8, -0.5).flatten()
        colsum[np.isinf(colsum)] = 0.
        colsum_diag = sp.diags(colsum)

        if mean_flag == False:
            return rowsum_diag*csr_mat*colsum_diag
        else:
            return rowsum_diag*csr_mat
        
    def mm(self, x, y): 
        return torch.sparse.mm(x, y)
    
    def get_knn_adj_mat(self, mm_embeddings):
        context_norm = mm_embeddings.div(torch.norm(mm_embeddings, p=2, dim=-1, keepdim=True))
        sim = torch.mm(context_norm, context_norm.transpose(1, 0))
        _, knn_ind = torch.topk(sim, self.knn_k, dim=-1)
        adj_size = sim.size()
        del sim
        # construct sparse adj
        indices0 = torch.arange(knn_ind.shape[0]).to(self.device)
        indices0 = torch.unsqueeze(indices0, 1)
        indices0 = indices0.expand(-1, self.knn_k)
        indices = torch.stack((torch.flatten(indices0), torch.flatten(knn_ind)), 0)
        # norm
        return indices, self.compute_normalized_laplacian(indices, adj_size)
    
    def get_extend_ui_mat_with_value(self,mm_embeddings,ui_graph,eta = 0.5,ori_mask=False):
        ui_dense = mm_embeddings.to_dense()
        context_norm = mm_embeddings.div(torch.norm(mm_embeddings, p=2, dim=-1, keepdim=True))
        mm_sim = torch.mm(context_norm, context_norm.transpose(1, 0))
        
        mm_mask = (mm_sim > 0).float()
        extend_ui_graph = torch.mm(ui_dense,mm_mask)
        if ori_mask:
            ui_mask = (~(ui_graph != 0)).float()
            extend_ui_graph = extend_ui_graph * ui_mask
        extend_ui_graph = (extend_ui_graph > 0).float()
        norm_ext_ui_graph = self.csr_norm(extend_ui_graph,mean_flag=False)
    
    def get_extend_ui_mat(self,mm_embeddings,ui_graph, topk=8):        
        context_norm = mm_embeddings.div(torch.norm(mm_embeddings, p=2, dim=-1, keepdim=True))
        sim = torch.mm(context_norm, context_norm.transpose(1, 0))
        adj_size = sim.size()
        
        _, knn_ind = torch.topk(sim, topk, dim=-1)
        knn_ind = knn_ind[:,1:]
        print(knn_ind[:10])
        indices0 = torch.arange(knn_ind.shape[0]).to(self.device)
        indices0 = torch.unsqueeze(indices0, 1)
        indices0 = indices0.expand(-1, topk-1)
        indices = torch.stack((torch.flatten(indices0), torch.flatten(knn_ind)), 0)
        ii_graph = torch.sparse.FloatTensor(indices, torch.ones_like(indices[0]).float(), adj_size)
        
        ui_graph_plus = torch.sparse.mm(ui_graph,ii_graph)
        ui_graph_plus = torch.sparse.FloatTensor(ui_graph_plus.indices(),
                                                 torch.ones_like(ui_graph_plus.values()).float(),
                                                 ui_graph_plus.size())
        ui_graph_plus = ui_graph_plus.coalesce()
        ui_graph_plus_norm = self.norm_extend_ui_graph(ui_graph_plus,ui_graph_plus.indices(),ui_graph_plus.size())
        return ui_graph_plus_norm
        
    def norm_extend_ui_graph(self,adj,indices,adj_size):
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        cols_inv_sqrt = r_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt
        return torch.sparse.FloatTensor(indices, values, adj_size)
        
    def compute_normalized_laplacian(self, indices, adj_size):
        adj = torch.sparse.FloatTensor(indices, torch.ones_like(indices[0]), adj_size)
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        cols_inv_sqrt = r_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt
        return torch.sparse.FloatTensor(indices, values, adj_size)

    def pre_epoch_processing(self):
        self.epoch_user_graph, self.user_weight_matrix = self.topk_sample(self.k)   #（19445,40) (19445,40)
        self.user_weight_matrix = self.user_weight_matrix.to(self.device)

    def pack_edge_index(self, inter_mat):
        rows = inter_mat.row
        cols = inter_mat.col + self.n_users
        # ndarray([598918, 2]) for ml-imdb
        return np.column_stack((rows, cols))

    def _gcn_pp(self, item_embed, user_embed, uig, iug, norm=False):
        if norm == True:
            item_res =  item_embed = F.normalize(item_embed)
            user_res =  user_embed = F.normalize(user_embed)
            for _ in range(2):
                user_agg = self.mm(uig, item_embed)
                item_agg = self.mm(iug, user_embed)
                item_embed = item_agg
                user_embed = user_agg
                
                item_res = item_res + item_embed
                user_res = user_res + user_embed
        
        else:
            item_res =  item_embed = F.normalize(item_embed)
            user_res =  user_embed = F.normalize(user_embed)
            for _ in range(2):
                user_agg = self.mm(uig, item_embed)
                item_agg = self.mm(iug, user_embed)
                item_embed = F.normalize(item_agg)
                user_embed = F.normalize(user_agg)
                
                item_res = item_res + item_embed
                user_res = user_res + user_embed
        
        return user_res, item_res
    
    def self_attention(self, t_feat, v_feat,pattern="train"):
        if pattern == "infer":
            t_feat = self.MLP_t(t_feat)
            v_feat = self.MLP_v(v_feat)
        
        # t_feat = self.MLP_t_a_2(t_feat)
        # v_feat = self.MLP_v_a_2(v_feat)
        t_score = torch.sigmoid(self.MLP_t_a(t_feat))
        v_score = torch.sigmoid(self.MLP_v_a(v_feat))
        if pattern == "infer":
            t_score = t_score.squeeze(-1)
            v_score = v_score.squeeze(-1)
        return t_score, v_score


    def forward(self):

        t_feat = self.t_feat
        v_feat = self.v_feat
        
        t_common, t_specific = self.text_decompose(t_feat)
        v_common, v_specific = self.image_decompose(v_feat)
        
        t_club_loss = self.text_club.compute_club_loss(t_common,t_specific)
        v_club_loss = self.image_club.compute_club_loss(v_common,v_specific,noise_scale = 2)
        club_loss = t_club_loss + v_club_loss
        rec_loss = self.cross_rec(t_common,t_specific,t_feat, v_common,v_specific, v_feat)

        self.t_common = t_common
        self.t_specific = t_specific
        
        self.v_common = v_common
        self.v_specific = v_specific
        
        modal_feat = v_common + t_specific + v_specific
       
        
        for _ in range(1):
            h = torch.sparse.mm(self.extend_ui_graph.transpose(0,1),user_embed)          
            user = torch.sparse.mm(self.extend_ui_graph,item_embed) 
            
            item_h = torch.sparse.mm(self.extend_ui_graph.transpose(0,1),user_embed) 
            user_h = torch.sparse.mm(self.extend_ui_graph,item_embed) 
            
            user_embed = user_embed + F.normalize(user_h)

        user_embed, ib_loss = self.FreeIB_model(user_embed)
        user_embed = torch.dropout(user_embed,p=0.3,train=self.training)

        self.u_embed = user_embed
        self.i_embed = item_embed

        return user_embed, item_embed, club_loss, 1 * rec_loss, 0.2 * ib_loss


    def _sparse_dropout(self, x, rate=0.0):
        noise_shape = x._nnz()                                       

        random_tensor = rate
        random_tensor += torch.rand(noise_shape).to(x.device)        
        dropout_mask = torch.floor(random_tensor).type(torch.bool)   
        i = x._indices()                                             
        v = x._values()                                              

        i = i[:, dropout_mask]                                       
        v = v[dropout_mask]                                          

        out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)   
        # return out * (1. / (1 - rate))                             
        return out

    def bpr_loss(self, interaction):
        user_embed, item_embed, club_loss, rec_loss, ib_loss = self.forward()
        user_nodes, pos_item_nodes, neg_item_nodes = interaction[0], interaction[1], interaction[2]

        user_embed = user_embed[user_nodes]
        pos_item_embed = item_embed[pos_item_nodes]
        neg_item_embed = item_embed[neg_item_nodes]
        pos_scores = torch.sum(torch.mul(user_embed, pos_item_embed), dim=1)
        neg_scores = torch.sum(torch.mul(user_embed, neg_item_embed), dim=1)

        regularizer = 1./2*(user_embed**2).sum() + 1./2*(pos_item_embed**2).sum() + 1./2*(neg_item_embed**2).sum()        
        regularizer = regularizer / self.batch_size

        maxi = F.logsigmoid(pos_scores - neg_scores)
        mf_loss = -torch.mean(maxi)

        emb_loss = 1e-4 * regularizer
        
        loss = mf_loss + emb_loss + 1e-4 * club_loss + 1 * rec_loss + 1 * ib_loss 
        return loss


    def _get_review_res(self):
        return self.review_score.cpu().numpy(), self.review_soft_list.cpu().numpy()

    def full_sort_predict(self, interaction):
        user_tensor, item_tensor,club_loss,rec_loss,ib_loss = self.forward()

        temp_user_tensor = user_tensor[interaction[0], :]
        score_matrix = torch.matmul(temp_user_tensor, item_tensor.t())
        return score_matrix

class Modal_Reviewer(torch.nn.Module):
    def __init__(self,num_user, num_item, dim_latent, n_interests, t_dim=None, v_dim =None, a_dim=None, dense_adj=None, select_value=0.3):
        super(Modal_Reviewer,self).__init__()
        self.num_user = num_user
        self.num_item = num_item
        self.dim_latent = dim_latent
        self.num_interests = n_interests
        self.select_value = select_value
        
        
        self.dense_adj_mask = (dense_adj != 0).float().transpose(1,0)
        self.dense_adj_bool = (self.dense_adj_mask.sum(-1) == 0.).float()
        self.t_dim = t_dim
        self.v_dim = v_dim
        self.a_dim = a_dim
            
    def forward(self,modal_feat,preference_list):
        preference = sum(preference_list) / len(preference_list)   
        # muti_preference = self._multi_interests(preference)        
        review_list = []
        review_soft_list = []
        for feat in modal_feat:                                         
            score_mat = torch.sigmoid(torch.matmul(feat,preference.transpose(1,0)))     
            score_mat = score_mat * self.dense_adj_mask
            feat_score = (score_mat.sum(dim=1)/(self.dense_adj_mask.sum(dim=1) + +self.dense_adj_bool)).unsqueeze(1)               
            review_list.append(feat_score)        
        return   torch.cat(review_list,dim=1)        
        
class User_Graph_sample(torch.nn.Module):
    def __init__(self, num_user, aggr_mode,dim_latent):
        super(User_Graph_sample, self).__init__()
        self.num_user = num_user
        self.dim_latent = dim_latent
        self.aggr_mode = aggr_mode

    def forward(self, features,user_graph,user_matrix):
        index = user_graph
        u_features = features[index]           #(19445,40,128)
        user_matrix = user_matrix.unsqueeze(1) #(19445,1,40)
        # pdb.set_trace()
        u_pre = torch.matmul(user_matrix,u_features)
        u_pre = u_pre.squeeze()
        return u_pre                           #(19445,128)

class GCN(torch.nn.Module):
    def __init__(self,datasets, batch_size, num_user, num_item, dim_id, aggr_mode, num_layer, has_id, dropout,
                 dim_latent=None,device = None,features=None):
        super(GCN, self).__init__()
        self.batch_size = batch_size
        self.num_user = num_user
        self.num_item = num_item
        self.datasets = datasets
        self.dim_id = dim_id
        self.dim_feat = features.size(1)
        self.dim_latent = dim_latent
        self.aggr_mode = aggr_mode
        self.num_layer = num_layer
        self.has_id = has_id
        self.dropout = dropout
        self.device = device

        if self.dim_latent:
            # self.MLP = nn.Linear(self.dim_feat, 4*self.dim_latent)                                  
            # self.MLP_1 = nn.Linear(4*self.dim_latent, self.dim_latent)                              
            self.conv_embed_1 = Base_gcn(self.dim_latent, self.dim_latent, aggr=self.aggr_mode)     
        else:
            self.conv_embed_1 = Base_gcn(self.dim_latent, self.dim_latent, aggr=self.aggr_mode)

    def forward(self,edge_index,features, preference):  
        temp_features = features
        x = torch.cat((preference, temp_features), dim=0).to(self.device)                          
        x = F.normalize(x).to(self.device)     
        h = self.conv_embed_1(x, edge_index)  # equation 1  [26495, 64], [2,237102] --> [26495, 64]
        h_1 = self.conv_embed_1(h, edge_index) # [26495, 64], [2,237102] --> [26495, 64]
        x_hat = x + h + h_1
    
        return x_hat

class Base_gcn(MessagePassing):
    def __init__(self, in_channels, out_channels, normalize=True, bias=True, aggr='add', **kwargs):
        super(Base_gcn, self).__init__(aggr=aggr, **kwargs)
        self.aggr = aggr                  #add
        self.in_channels = in_channels    #64
        self.out_channels = out_channels  #64

    def forward(self, x, edge_index, size=None):
        if size is None:
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        x = x.unsqueeze(-1) if x.dim() == 1 else x  #[26495, 64]
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_j, edge_index, size):
        if self.aggr == 'add':
            # pdb.set_trace()
            row, col = edge_index
            deg = degree(row, size[0], dtype=x_j.dtype)
            deg_inv_sqrt = deg.pow(-0.5)
            norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
            return norm.view(-1, 1) * x_j
        return x_j

    def update(self, aggr_out):
        return aggr_out

    def __repr(self):
        return '{}({},{})'.format(self.__class__.__name__, self.in_channels, self.out_channels)

class Linears(torch.nn.Module):
    def __init__(self,inp_dim, out_dim, ln=True) :
        super(Linears, self).__init__()
        self.inp_dim = inp_dim
        self.out_dim = out_dim
        self.ln = ln
        self.layer = nn.Linear(self.inp_dim, self.out_dim)
        nn.init.xavier_uniform_(self.layer.weight)
        if self.ln:
            self.LN = nn.LayerNorm(self.out_dim)
    def forward(self,x):
        x = self.layer(x)
        if self.ln:
            x = self.LN(x) + x
        return x
    
class Encoder(torch.nn.Module):
    def __init__(self,inp_dim, out_dim,channel_list):
        super(Encoder,self).__init__()
        self.inp_dim = inp_dim
        self.out_dim = out_dim
        self.channel_list = channel_list
        self.channel_list.insert(0, self.inp_dim)
        self.layers = [Linears(self.channel_list[i],self.channel_list[i+1],ln=True) for i in range(len(self.channel_list)-1)]
        self.layers.append(Linears(self.channel_list[-1], self.out_dim))
        self.model = nn.Sequential(*self.layers)
        
    def forward(self, x):
        out = self.model(x)
        return out

class Decoder(torch.nn.Module):
    def __init__(self,inp_dim, out_dim, channel_list):
        super(Decoder,self).__init__()
        self.inp_dim = inp_dim
        self.out_dim = out_dim
        self.channel_list = channel_list
        self.channel_list.insert(0, self.inp_dim)
        self.layers = [Linears(self.channel_list[i],self.channel_list[i+1], ln=True) for i in range(len(self.channel_list)-1)]
        # self.layers.append(nn.Linear(self.channel_list[-1], self.out_dim))
        self.layers.append(Linears(self.channel_list[-1], self.out_dim))
        self.model = nn.Sequential(*self.layers)
        
    def forward(self, x):
        out = self.model(x)
        return out

class Cookbook(torch.nn.Module):
    def __init__(self, num_codebook_vectors, code_dim, beta=0.1):
        super(Cookbook, self).__init__()
        self.num_codebook_vectors = num_codebook_vectors
        self.code_dim = code_dim
        self.beta = beta
        self.embedding = nn.Embedding(self.num_codebook_vectors, self.code_dim)
        nn.init.xavier_normal_(self.embedding.weight, gain=1.0)
    
    def forward(self, z):
        z_r = z.view(-1, self.code_dim).contiguous()

        d_0 = torch.sum(z_r**2, dim=1, keepdim=True)
        d_1 = torch.sum(self.embedding.weight**2, dim=1)
        d_2 = 2*(torch.matmul(z_r, self.embedding.weight.t()))
        d = d_0 + d_1 - d_2
 
        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)

        code_loss = torch.mean((z_q.detach() - z)**2) + self.beta * torch.mean((z_q - z.detach())**2)

        z_q = z + (z_q - z).detach() 
        return z_q, code_loss

class ModalDecompose(nn.Module):
    def __init__(self,inp_dim,device,n_user,n_item):
        super(ModalDecompose,self).__init__()
        # self.decompose_layer = nn.Linear(inp_dim,inp_dim * 2)
        # self.common_layer = nn.Sequential(nn.Linear(inp_dim, inp_dim),
        #                                   nn.Tanh())
        # self.specific_layer = nn.Sequential(nn.Linear(inp_dim, inp_dim),
        #                                     nn.Tanh())
        
        self.common_layer = nn.Parameter(nn.init.xavier_normal_(torch.tensor(
            np.random.randn(inp_dim, inp_dim), dtype=torch.float32, requires_grad=True),
            gain=1).to(device)) 
        self.specific_layer = nn.Parameter(nn.init.xavier_normal_(torch.tensor(
            np.random.randn(inp_dim, inp_dim), dtype=torch.float32, requires_grad=True),
            gain=1).to(device)) 

        self.decompose_layer = nn.Sequential(nn.Linear(inp_dim, inp_dim),
                                            nn.LayerNorm(inp_dim),
                                            nn.Linear(inp_dim, inp_dim))

        self.decouple_com = nn.Linear(inp_dim,inp_dim)
        self.decouple_spe = nn.Linear(inp_dim,inp_dim)
        self.decouple_com_2 = nn.Linear(inp_dim,inp_dim)
        self.decouple_spe_2 = nn.Linear(inp_dim,inp_dim)
        self.decouple_com_3 = nn.Linear(inp_dim,inp_dim)
        self.decouple_spe_3 = nn.Linear(inp_dim,inp_dim)
        self.ln = nn.LayerNorm(inp_dim)

        # self.decouple_spe = nn.Parameter(nn.init.xavier_normal_(torch.tensor(
        #     np.random.randn(n_item, inp_dim), dtype=torch.float32, requires_grad=True),
        #     gain=1).to(device)) 
        # self.decompose_layer_3 = nn.Linear(inp_dim,inp_dim * 2,bias=False)

    def gumbel_process(self,action_prob, tau=1, dim=-1, hard=True):
        gumbels = (
            -torch.empty_like(action_prob, memory_format=torch.legacy_contiguous_format).exponential_().log()
        )  # ~Gumbel(0,1)
        # gumbels = (action_prob + gumbels) / tau  # ~Gumbel(logits,tau)
        randoms = torch.rand_like(action_prob)
        gumbels = (action_prob + 0.1 * gumbels)/tau
        # gumbels = action_prob / tau
        y_soft = gumbels.softmax(dim)

        if hard:
            index = y_soft.max(dim, keepdim=True)[1]
            y_hard = torch.zeros_like(action_prob, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
            ret = y_hard - y_soft.detach() + y_soft
        else:
            ret = y_soft
        return ret
    # def decompose(self,modal_feature):
    #     hidden = self.decompose_layer(modal_feature).reshape(modal_feature.shape[0],-1,modal_feature.shape[-1])
    #     common_feature, specific_feature = hidden[:,0], hidden[:,-1]
    # #     return common_feature, specific_feature
    def decompose(self,modal_feature):
        common_feature, specific_feature = self.common_layer(modal_feature),self.specific_layer(modal_feature)
        return torch.dropout(torch.tanh(common_feature),p=0.3,train = self.training), torch.dropout(torch.tanh(specific_feature),p=0.3,train=self.training)
    # def decompose(self,modal_feature):
        # common_feature, specific_feature = torch.matmul(modal_feature, self.common_layer), torch.matmul(modal_feature,self.specific_layer)
        # return common_feature, specific_feature
        # return torch.dropout(torch.tanh(common_feature),p=0.3,train = self.training), torch.dropout(torch.tanh(specific_feature),p=0.3,train=self.training)
    def decompose_2(self,modal_feature):
        modal_score = torch.sigmoid(self.decompose_layer(modal_feature))
        gumbels = (
                    -torch.empty_like(modal_score, memory_format=torch.legacy_contiguous_format).exponential_().log()
                )
        modal_score = modal_score + 0.1 * gumbels
        com_multi_hot = (modal_score > 0.5).float()
        com_multi_hot = com_multi_hot - modal_score.detach() + modal_score
        spe_multi_hot = (1-com_multi_hot)
        modal_common, modal_specific = com_multi_hot * modal_feature, spe_multi_hot * modal_feature
        return torch.dropout(modal_common,p=0.0,train = self.training), torch.dropout(modal_specific,p=0.0,train=self.training)
    
    def decompose_4(self,modal_feature):
        modal_common, modal_specific =  self.decouple_com(modal_feature), self.decouple_spe(modal_feature)
        modal_common, modal_specific = self.ln(modal_common) + modal_common, self.ln(modal_specific) + modal_specific
        
        modal_common, modal_specific =  self.decouple_com_2(modal_common), self.decouple_spe_2(modal_specific)
        modal_common, modal_specific = self.ln(modal_common) + modal_common, self.ln(modal_specific) + modal_specific
        
        modal_common, modal_specific =  self.decouple_com_3(modal_common), self.decouple_spe_3(modal_specific)
        modal_common = torch.dropout(modal_common,p=0.4,train=self.training)
        modal_specific = torch.dropout(modal_specific,p=0.4,train=self.training)
        return modal_common, modal_specific
    # def decompose_3(self,modal_feature):
    #     modal_score = self.decompose_layer_3(modal_feature)
    #     modal_score = modal_score.view(modal_feature.shape[0],modal_feature.shape[1],-1)
    #     # modal_score = torch.softmax(modal_score,dim=-1)[:,:,0]
    #     hard_score = self.gumbel_process(modal_score)
    #     modal_common, modal_specific = hard_score[:,:,0] * modal_feature, hard_score[:,:,1] * modal_feature
    #     return torch.dropout(modal_common,p=0.2,train = self.training), torch.dropout(modal_specific,p=0.4,train=self.training)
    
    
    def forward(self,modal_feature):
        return self.decompose(modal_feature)
        #return self.decompose_4(modal_feature)

class CLUB(nn.Module):
    def __init__(self,inp_dim):
        super(CLUB,self).__init__()
        self.mu_layer = nn.Sequential(nn.Linear(inp_dim, inp_dim),
                                      nn.ReLU())
        self.logvar_layer = nn.Sequential(nn.Linear(inp_dim, inp_dim),
                                          nn.ReLU())
    
    def get_mu_logvar(self,x_com):
        mu = self.mu_layer(x_com)
        logvar = self.logvar_layer(x_com)
        return mu, logvar
        
    
    def loglikeli(self, x_com, x_spe):
        mu, logvar = self.get_mu_logvar(x_com)
        return (-(mu - x_spe)**2 /logvar.exp()-logvar).sum(dim=1).mean(dim=0)
    
    def eval_mi(self, x_com, x_spe):
        mu, logvar = self.get_mu_logvar(x_com)
        
        sample_size = x_com.shape[0]
        #random_index = torch.randint(sample_size, (sample_size,)).long()
        random_index = torch.randperm(sample_size).long()
        
        positive = - (mu - x_spe)**2 / logvar.exp()
        negative = - (mu - x_spe[random_index])**2 / logvar.exp()
        upper_bound = (positive.sum(dim = -1) - negative.sum(dim = -1)).mean()
        return upper_bound/2
    
    def cal_club_loss(self, x_com, x_spe):
        return -self.loglikeli(x_com, x_spe)
        # return self.eval_mi(x_com,x_spe)

    def compute_club_loss(self, x_com, x_spe,noise_scale = 1):
        mu, logvar = self.get_mu_logvar(x_com)
        indices = torch.randperm(mu.shape[0])
        logvar = torch.clamp(logvar, min=-10, max=10)
        
        log_likelihood = -((x_spe - mu)**2 / (logvar.exp() + 1) + logvar)

        log_likelihood = log_likelihood.sum(dim=1).mean(dim=0)
        shuffled_x_spe = x_spe[torch.randperm(x_spe.size(0))]
        shuffled_x_spe = shuffled_x_spe[:, torch.randperm(x_spe.size(1))]

        noise = torch.rand_like(x_spe) * 2 * noise_scale - noise_scale
        shuffled_x_spe = shuffled_x_spe + noise
        negative_log_likelihood = -(((shuffled_x_spe - mu)**2 / (logvar.exp() + 1) + logvar).sum(dim=1).mean(dim=0))
    
        club_loss = log_likelihood - 2 * negative_log_likelihood
    
        total_loss = club_loss 

        return total_loss 
    
class Corss_Rec(nn.Module):
    def __init__(self,common_dim, specific_dim, modal_dim):
        super(Corss_Rec,self).__init__()
        self.text_linear = nn.Linear(common_dim,modal_dim)
        self.image_linear = nn.Linear(common_dim,modal_dim)
        self.ln = nn.LayerNorm(common_dim)
        
        # self.text_rec = nn.Linear(common_dim,modal_dim)
        # self.image_rec = nn.Linear(common_dim,modal_dim)
        self.text_rec = nn.Sequential(nn.Linear(common_dim,modal_dim),
                                      nn.Tanh(),
                                      nn.Linear(common_dim,modal_dim))
        
        self.image_rec = nn.Sequential(nn.Linear(common_dim,modal_dim),
                                       nn.Tanh(),
                                       nn.Linear(common_dim,modal_dim))
        self.rec_loss_func = nn.MSELoss() 
        
    def cal_rec_loss(self,rec_feature, ori_feature):
        rec_loss = self.rec_loss_func(rec_feature,ori_feature)
        return rec_loss
    
    def forward(self,text_common, text_specific, text_origin, image_common, image_specific, image_origin):
        tgt_text = image_common + text_specific
        text_rec_loss = self.cal_rec_loss(tgt_text, text_origin) 
        tgt_image = text_common + image_specific
        image_rec_loss = self.cal_rec_loss(tgt_image,image_origin)
        rec_loss = text_rec_loss + image_rec_loss 
        cosine_similarity = torch.nn.functional.cosine_similarity(text_common, image_common, dim=1).mean()

        cosine_loss = 1 - cosine_similarity
        return rec_loss + cosine_loss
    
class FreeIBModel(nn.Module):
    def __init__(self, inp_dim, preference, n_user,devide):
        super(FreeIBModel, self).__init__()
        self.linear = nn.Linear(inp_dim, inp_dim)
        self.ln = nn.LayerNorm(inp_dim)
        self.c_layer = nn.Sequential(nn.Linear(inp_dim, inp_dim),
                                     nn.Tanh(),
                                     nn.Linear(inp_dim, inp_dim),
                                     nn.Sigmoid())
    
    def forward(self, preference):   
        hidden = self.linear(preference)
        hidden = self.ln(hidden) + hidden
        c_value = self.c_layer(hidden)
        refined_preference = c_value  * preference
        c_mu = torch.mean(c_value.mean(dim=-1),dim=0)
        c_var = torch.std(c_value,dim=-1).mean(dim=0)
        ob_loss = c_mu - c_var
        return refined_preference, ob_loss
    

class IBModel(nn.Module):
    def __init__(self, inp_dim, preference,n_user,device):
        super(IBModel, self).__init__()
        self.mu_layer = nn.Linear(inp_dim, inp_dim)
        self.logvar_layer = nn.Linear(inp_dim, inp_dim)
        self.ln = nn.LayerNorm(inp_dim)
        self.user_id_preference = preference
        self.c_layer = nn.Linear(inp_dim, inp_dim)
        
        self.mu_prior = torch.zeros(n_user,inp_dim).to(device)
        self.std_prior = torch.ones(n_user,inp_dim).to(device)
        # self.mu_layer = nn.Parameter(nn.init.xavier_normal_(torch.tensor(
        # np.random.randn(inp_dim, inp_dim), dtype=torch.float32, requires_grad=True),
        # gain=1))
    def get_mu_var(self,x,a=-0.2,b=0.2):
        c = self.c_layer(x + self.user_id_preference)
        
        mu = self.mu_layer(x)
        # mu = x
        logvar = self.logvar_layer(x)
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std) 
        # epsilon = torch.rand_like(x) * (b-a) + a
        # std = torch.exp(0.5 * logvar) 
        # z = mu + std * torch.randn_like(std)
        z = x * torch.sigmoid(c)
        random_z = z + 0.01 * epsilon
        
        # z = mu 
        return z,random_z, mu, std, logvar
    
    def cal_ib_loss(self,x):
        z,random_z, mu, std, logvar = self.get_mu_var(x)
        kl_loss = self.kl_divergence(mu, std)
        return kl_loss,z
    
    def kl_loss(self,z_u,x_u):
        indices = torch.randperm(z_u.shape[0])[:1024]
        z_u = z_u[indices]
        x_u = x_u[indices]
        xx = torch.mm(z_u, z_u.t())
        yy = torch.mm(x_u, x_u.t())
        xy = torch.mm(z_u, x_u.t())
        mmd = xx.mean() + yy.mean() - 2 * xy.mean()
        return mmd
    
    def kl_uniform_loss(self,z, a=-0.2, b=0.2):
        prior_density = 1 / (b - a)  
        posterior_density = torch.ones_like(z) / (b - a)
        kl = posterior_density * torch.log(posterior_density / prior_density)
        return kl.sum(dim=1).mean()
    
    def kl_divergence(self, mu, std):

        var = std ** 2
        var_prior = self.std_prior ** 2
        kl = 0.5 * (torch.log(var_prior / var) + (var + (mu - self.mu_prior) ** 2) / var_prior - 1)
        return kl.sum(dim=1).mean()
