# coding: utf-8


r"""
################################
"""

import os
import itertools
from pyexpat import model
import torch
import torch.optim as optim
from torch.nn.utils.clip_grad import clip_grad_norm_
import numpy as np
import matplotlib.pyplot as plt

import pickle as pkl

from time import time
from logging import getLogger

from utils.utils import get_local_time, early_stopping, dict2str
from utils.topk_evaluator import TopKEvaluator
from utils.dataloader import TSDataset
from models.mm_adapter_v2 import VQGANTrainer 
from torch.utils.data import DataLoader

class AbstractTrainer(object):
    r"""Trainer Class is used to manage the training and evaluation processes of recommender system models.
    AbstractTrainer is an abstract class in which the fit() and evaluate() method should be implemented according
    to different training and evaluation strategies.
    """

    def __init__(self, config, model):
        self.config = config
        self.model = model

    def fit(self, train_data):
        r"""Train the model based on the train data.

        """
        raise NotImplementedError('Method [next] should be implemented.')

    def evaluate(self, eval_data):
        r"""Evaluate the model based on the eval data.

        """

        raise NotImplementedError('Method [next] should be implemented.')


class Trainer(AbstractTrainer):
    r"""The basic Trainer for basic training and evaluation strategies in recommender systems. This class defines common
    functions for training and evaluation processes of most recommender system models, including fit(), evaluate(),
   and some other features helpful for model training and evaluation.

    Generally speaking, this class can serve most recommender system models, If the training process of the model is to
    simply optimize a single loss without involving any complex training strategies, such as adversarial learning,
    pre-training and so on.

    Initializing the Trainer needs two parameters: `config` and `model`. `config` records the parameters information
    for controlling training and evaluation, such as `learning_rate`, `epochs`, `eval_step` and so on.
    More information can be found in [placeholder]. `model` is the instantiated object of a Model Class.

    """

    def __init__(self, config, model):
        super(Trainer, self).__init__(config, model)

        self.logger = getLogger()
        self.learner = config['learner']                       #adam
        self.learning_rate = config['learning_rate']
        self.epochs = config['epochs']                         #1000
        self.eval_step = min(config['eval_step'], self.epochs) #1
        self.stopping_step = config['stopping_step']           #20
        self.clip_grad_norm = config['clip_grad_norm']         #None
        self.valid_metric = config['valid_metric'].lower()     
        self.valid_metric_bigger = config['valid_metric_bigger'] #True
        self.test_batch_size = config['eval_batch_size']       #4096
        self.device = config['device']                         

        self.start_epoch = 0
        self.cur_step = 0
        self.best_valid_score = -1
        self.best_valid_result = None
        self.best_test_upon_valid = None
        self.train_loss_dict = dict()
        self.optimizer = self._build_optimizer()

        fac = lambda epoch: 1.0 - (epoch / 70)
        lr_scheduler = config['learning_rate_scheduler']        # check zero?
        # fac = lambda epoch: lr_scheduler[0] ** (epoch / lr_scheduler[1])
        scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=fac)
        self.lr_scheduler = scheduler

        self.eval_type = config['eval_type']
        self.evaluator = TopKEvaluator(config)

        self.save_flag = True
        # self.item_tensor = None
        # self.tot_item_num = None

        # # self.t_feat,self.v_feat = self.model.t_feat.cpu().numpy(), self.model.v_feat.cpu().numpy()
        # self.t_modal_score = None
        # self.v_modal_score = None
        # self.threshold = 0.4
        # self.text_threshold = 0.5
        # self.image_threshold = 0.5
        # self.threshold_low = 0.4
        # self.group_index = None
        # self.group_data = None
        
    # def _get_ts_data(self):
    #     t_v_h = np.where((self.t_modal_score >= self.text_threshold) & (self.v_modal_score >= self.image_threshold))[0]
    #     print("both high number:", t_v_h.shape[0])
    #     t_v_h_feature = np.concatenate([self.t_feat[t_v_h], self.v_feat[t_v_h]],axis=0)
    #     t_v_h_label = np.concatenate([self.v_feat[t_v_h], self.t_feat[t_v_h]],axis=0)
    #     t_v_h_pattern = np.concatenate([np.zeros(t_v_h.shape, dtype=int), np.ones(t_v_h.shape, dtype=int)],axis=0)
        
    #     # ### src: image; tgt:text
    #     # t_v_h_feature = self.v_feat[t_v_h]
    #     # t_v_h_label = self.t_feat[t_v_h]
    #     # t_v_h_pattern = np.ones(t_v_h.shape, dtype=int)
    #     # ### src: text; tgt:image
    #     # t_v_h_feature = self.t_feat[t_v_h]
    #     # t_v_h_label = self.v_feat[t_v_h]
    #     # t_v_h_pattern = np.zeros(t_v_h.shape, dtype=int)
        
        
        
        
    #     t_v_h_data = [t_v_h_feature, t_v_h_pattern, t_v_h_label]
    #     pkl.dump(t_v_h_data, open("t_v_h_data.pkl","wb"))
        
    #     th_vl = np.where((self.t_modal_score >= self.text_threshold) & (self.v_modal_score < self.image_threshold))[0]
    #     print("text high and vison low number:", th_vl.shape[0])
    #     th_vl_data = [self.t_feat[th_vl], np.zeros(th_vl.shape)]
        
    #     # tl_vh = np.where((self.t_modal_score < self.threshold) & (self.v_modal_score >= self.threshold))[0] #
    #     tl_vh = np.where((self.t_modal_score < self.text_threshold) & (self.v_modal_score >= self.image_threshold))[0] #
    #     print("vison high and text low number:", tl_vh.shape[0])
    #     tl_vh_data = [self.v_feat[tl_vh], np.ones(tl_vh.shape)]
        
    #     group_index = [t_v_h, th_vl, tl_vh]
    #     group_data = [t_v_h_data, th_vl_data, tl_vh_data]
        
    #     return group_index, group_data

    def _build_optimizer(self):
        r"""Init the Optimizer

        Returns:
            torch.optim: the optimizer
        """
        if self.learner.lower() == 'adam':
            optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        elif self.learner.lower() == 'sgd':
            optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate)
        elif self.learner.lower() == 'adagrad':
            optimizer = optim.Adagrad(self.model.parameters(), lr=self.learning_rate)
        elif self.learner.lower() == 'rmsprop':
            optimizer = optim.RMSprop(self.model.parameters(), lr=self.learning_rate)
        else:
            self.logger.warning('Received unrecognized optimizer, set default Adam optimizer')
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return optimizer

    def _train_epoch(self, train_data, epoch_idx, loss_func=None):
        r"""Train the model in an epoch

        Args:
            train_data (DataLoader): The train data.
            epoch_idx (int): The current epoch id.
            loss_func (function): The loss function of :attr:`model`. If it is ``None``, the loss function will be
                :attr:`self.model.calculate_loss`. Defaults to ``None``.

        Returns:
            float/tuple: The sum of loss returned by all batches in this epoch. If the loss in each batch contains
            multiple parts and the model return these multiple parts loss instead of the sum of loss, It will return a
            tuple which includes the sum of loss in each part.
        """
        self.model.train()
        # loss_func = loss_func or self.model.calculate_loss
        loss_func = self.model.bpr_loss
        total_loss = None
        loss_batches = []
        for batch_idx, interaction in enumerate(train_data):
            
            losses = loss_func(interaction)
            if isinstance(losses, tuple):
                loss = sum(losses)
                loss_tuple = tuple(per_loss.item() for per_loss in losses)
                total_loss = loss_tuple if total_loss is None else tuple(map(sum, zip(total_loss, loss_tuple)))
            else:
                loss = losses
                total_loss = losses.item() if total_loss is None else total_loss + losses.item()
            self._check_nan(loss)
            loss.backward()
            if self.clip_grad_norm:
                clip_grad_norm_(self.model.parameters(), **self.clip_grad_norm)
            self.optimizer.step()
            
            loss_batches.append(loss.detach())
            # for test
            #if batch_idx == 0:
            #    break
            self.optimizer.zero_grad()
        return total_loss, loss_batches

    def _valid_epoch(self, valid_data, is_test=False, idx=0):
        r"""Valid the model with valid data

        Args:
            valid_data (DataLoader): the valid data

        Returns:
            float: valid score
            dict: valid result
        """
        valid_result = self.evaluate(valid_data,is_test,idx)
        valid_score = valid_result[self.valid_metric] if self.valid_metric else valid_result['NDCG@20']
        return valid_score, valid_result

    def _check_nan(self, loss):
        if torch.isnan(loss):
            raise ValueError('Training loss is nan')

    def _generate_train_loss_output(self, epoch_idx, s_time, e_time, losses):
        train_loss_output = 'epoch %d training [time: %.2fs, ' % (epoch_idx, e_time - s_time)
        if isinstance(losses, tuple):
            train_loss_output = ', '.join('train_loss%d: %.4f' % (idx + 1, loss) for idx, loss in enumerate(losses))
        else:
            train_loss_output += 'train loss: %.4f' % losses
        return train_loss_output + ']'
    
    # def _obtain_modal_score(self, config, epoch_idx):
    #     self.model.eval()
    #     root_path = './data/' + config['dataset'] + "/modal_score"
    #     root_path_2 = './data/' + config['dataset'] + "/modal_score_and_embed_g"
    #     if not os.path.exists(root_path):
    #         os.mkdir(root_path)
    #     if not os.path.exists(root_path_2):
    #         os.mkdir(root_path_2)
    #     if epoch_idx%1 == 0 and epoch_idx < 100 and epoch_idx != 0:
    #         t_score,v_score = self.model.t_score.detach().cpu().numpy(), self.model.v_score.detach().cpu().numpy()

    #         user_tensor, item_tensor = self.model.forward()
    #         user_numpy, item_numpy = user_tensor.detach().cpu().numpy(), item_tensor.detach().cpu().numpy()
    #         del user_tensor, item_tensor, user_numpy, item_numpy
    #         torch.cuda.empty_cache()
            
    # def _build_ts_dataloader(self, config):
    #     self.t_modal_score,self.v_modal_score = self.model.t_score.detach().cpu().numpy(), self.model.v_score.detach().cpu().numpy()
    #     self.group_index, self.group_data = self._get_ts_data()
    #     ts_dataset = TSDataset(config, self.group_data[0], mode = "train")
    #     data_loader = DataLoader(ts_dataset, batch_size=128, shuffle=True, num_workers=0)
        
    #     th_vl_dataset = TSDataset(config, self.group_data[1], mode = "valid")  
    #     th_vl_dataloader = DataLoader(th_vl_dataset, batch_size=128, shuffle=False, num_workers=0)
    #     tl_vh_dataset = TSDataset(config, self.group_data[-1], mode = "valid")
    #     tl_vh_dataloader = DataLoader(tl_vh_dataset, batch_size=128, shuffle=False, num_workers=0)
    #     return data_loader, th_vl_dataloader, tl_vh_dataloader
    
    # def _replace_modal_feat(self, new_image_feat, new_image_index, new_text_feat, new_text_index):
    #     pkl.dump(self.v_feat[new_image_index],open("or_image_feat.pkl","wb"))
    #     pkl.dump(self.t_feat[new_text_index],open("or_text_feat.pkl","wb"))

    #     pkl.dump(new_image_feat,open("new_image_feat.pkl","wb"))
    #     pkl.dump(new_text_feat,open("new_text_feat.pkl","wb"))
    #     # self.v_feat[th_vl_index] = th_vl_feat
    #     # self.t_feat[tl_vh_index] = tl_vh_feat
    #     # print("replace number: ", tl_vh_index.shape)
    #     # self.t_feat[tl_vh_index] = self.t_feat[tl_vh_index] + tl_vh_feat
    #     # self.t_feat[tl_vh_index] = np.zeros(tl_vh_feat.shape)
    #     self.t_feat[new_text_index] =   new_text_feat
    #     self.v_feat[new_image_index] =  new_image_feat
    
    # def check_vq_embed(self,image_pred, image_index, text_pred, text_index):
    #     self.model.eval()
        
    #     image_pred_tensor = torch.tensor(image_pred, dtype=torch.float32).to(self.device)
    #     text_pred_tensor = torch.tensor(text_pred, dtype=torch.float32).to(self.device)
        
    #     image_index_tensor = torch.tensor(image_index, dtype=torch.int32).to(self.device)
    #     text_index_tensor = torch.tensor(text_index, dtype=torch.int32).to(self.device)
        
    #     image_score = self.model.review_modal([image_pred_tensor], index=image_index_tensor, check_pattern="image").squeeze(-1)
    #     image_score = image_score.detach().cpu().numpy()
    #     text_score = self.model.review_modal([text_pred_tensor], index = text_index_tensor, check_pattern="text").squeeze(-1)
    #     text_score = text_score.detach().cpu().numpy()
    #     # tl_vh_score, th_vl_score = self.model.self_attention(tl_vh_tensor, th_vl_tensor,pattern="infer")

    #     # th_vl_score_bool = th_vl_score > (self.threshold + 0.15)
    #     # tl_vh_score_bool = tl_vh_score > (self.threshold + 0.15)
    #     image_score_bool = image_score > self.image_threshold
    #     text_score_bool = text_score > self.text_threshold
        
    #     image_res_score = image_score[image_score_bool]
    #     text_res_score = text_score[text_score_bool]
        
        
    #     new_image_index = image_index[image_score_bool]
    #     new_text_index = text_index[text_score_bool]
        
    #     new_image_feat = image_pred[image_score_bool]
    #     new_text_feat = text_pred[text_score_bool]
        
    #     return [new_image_index, new_text_index], [new_image_feat, new_text_feat], [image_res_score,text_res_score]

    def save_embed(self, config, epoch_idx,model_name = "Cleaner"):
        user_embed, item_embed, club_loss, rec_loss, ib_loss = self.model.forward()
        root_path = './data/' + config['dataset'] + "/" + model_name + "_embed"
        if not os.path.exists(root_path):
            os.mkdir(root_path)
            
        user_numpy = user_embed.detach().cpu().numpy()
        item_numpy = item_embed.detach().cpu().numpy()
        np.save(root_path+ "/user_embed" + str(epoch_idx) + ".npy", user_numpy)
        np.save(root_path+ "/item_embed" + str(epoch_idx) + ".npy", item_numpy)
        del user_embed, item_embed, user_numpy, item_numpy

    def save_com_spe_embed(self, config, epoch_idx,model_name = "Cleaner"):
        t_common, v_common, t_specific, v_specific = self.model.t_common, self.model.v_common, self.model.t_specific, self.model.v_specific
        root_path = './data/' + config['dataset'] + "/" + model_name + "_com_spe"
        if not os.path.exists(root_path):
            os.mkdir(root_path)
        
        t_common = t_common.detach().cpu().numpy()
        v_common = v_common.detach().cpu().numpy()
        t_specific = t_specific.detach().cpu().numpy()
        v_specific = v_specific.detach().cpu().numpy()
        
        common_feat = [t_common, v_common]
        specific_feat = [t_specific, v_specific]
        # user_numpy = user_embed.detach().cpu().numpy()
        # item_numpy = item_embed.detach().cpu().numpy()
        pkl.dump(common_feat, open(root_path+ "/common" + str(epoch_idx) + ".pkl","wb"))
        pkl.dump(specific_feat, open(root_path+ "/specific" + str(epoch_idx) + ".pkl","wb"))
        # np.save(root_path+ "/user_embed" + str(epoch_idx) + ".npy", user_numpy)
        # np.save(root_path+ "/item_embed" + str(epoch_idx) + ".npy", item_numpy)
        del common_feat, specific_feat, t_common, v_common, t_specific, v_specific
        
        
        
        
    def fit(self,config, train_data, valid_data=None, test_data=None, saved=False, verbose=True):
        r"""Train the model based on the train data and the valid data.

        Args:
            train_data (DataLoader): the train data
            valid_data (DataLoader, optional): the valid data, default: None.
                                               If it's None, the early_stopping is invalid.
            test_data (DataLoader, optional): None
            verbose (bool, optional): whether to write training and evaluation information to logger, default: True
            saved (bool, optional): whether to save the model parameters, default: True
 
        Returns:
             (float, dict): best valid score and best valid result. If valid_data is None, it returns (-1, None)
        """
        best_metric = {"recall@10":0, "recall@20":0,  "ndcg@10":0, "ndcg@20":0}
        best_epoch  = {"recall@10":0, "recall@20":0,  "ndcg@10":0, "ndcg@20":0}
        # vq_config = {"ec_inp_dim":384, "ec_out_dim":1024, "tgt_dim":384, "prompt_dim": 256, "prompt_vocab_size": 2,
        #             "code_book_num_vector": 4096, "code_dim": 32, "beta1": 0.5, "beta2": 0.9}
        # vq_config["device"] = config["device"]
        # vq_config["vq_lr"] = 5e-4
        # vq_config["dis_lr"] = 5e-5
        # VQGAN_model = VQGANTrainer(vq_config)
        stop_flag = 0
        # ts_start_epoch = 5
        # ts_cnt = 0
        for epoch_idx in range(self.start_epoch, self.epochs):
            training_start_time = time()
            # self.model.pre_epoch_processing()
            train_loss, _ = self._train_epoch(train_data, epoch_idx)
            # if epoch_idx % 6 == 0 and epoch_idx!=0 and ts_cnt < 10:
            # # if epoch_idx == 1:
            # # if epoch_idx == 1:
            #     print("beging replace >>>>>>>>>>>>>>")
            #     self.t_feat,self.v_feat = self.model.t_feat.detach().cpu().numpy(), self.model.v_feat.detach().cpu().numpy()
            #     ts_dataloader, th_vl_dataloader, tl_vh_dataloader = self._build_ts_dataloader(config)
            #     if ts_cnt>0:
            #         config["TS_epoch"] = 20
            #     VQGAN_model.train(ts_dataloader, config)
            #     image_pred = VQGAN_model.inference(th_vl_dataloader, data_type="th_vl_data")
            #     text_pred = VQGAN_model.inference(tl_vh_dataloader, data_type="tl_vh_data")

            #     image_index, text_index = self.group_index[1], self.group_index[2]
            #     index, new_feat, res_scores = self.check_vq_embed(image_pred, image_index, text_pred, text_index)
            #     new_image_index, new_text_index = index[0], index[1]
            #     new_image_feat, new_text_feat = new_feat[0], new_feat[1]
            #     image_score, text_score = res_scores[0], res_scores[1]

                
            #     self._replace_modal_feat(new_image_feat, new_image_index, new_text_feat, new_text_index)
                
            #     # self._replace_modal_feat(th_vl_pred, th_vl_index, tl_vh_pred, tl_vh_index)
            #     self.model.update_modal_feat(self.t_feat, self.v_feat)
            #     ts_cnt += 1

            self.lr_scheduler.step()

            self.train_loss_dict[epoch_idx] = sum(train_loss) if isinstance(train_loss, tuple) else train_loss
            training_end_time = time()
            train_loss_output = \
                self._generate_train_loss_output(epoch_idx, training_start_time, training_end_time, train_loss)

            if verbose:
                self.logger.info(train_loss_output)


            if (epoch_idx + 1) % self.eval_step == 0:

                valid_score, valid_result = self._valid_epoch(valid_data)

                _, test_result = self._valid_epoch(test_data, False, epoch_idx)
                if verbose:

                    self.logger.info('epoch ' + str(epoch_idx) + ' test result: \n' + '\t' + dict2str(test_result))
                for k in best_metric.keys():
                    if test_result[k] > best_metric[k]:
                        best_metric[k] = test_result[k]
                        self.best_test_upon_valid = test_result
                        self.best_valid_result = valid_result
                        best_epoch[k] = epoch_idx 
                        stop_flag = 0
                        
                        if self.save_flag and epoch_idx > 100:
                            self.save_embed(config, epoch_idx, model_name = "Cleaner wo PR")
                            # if epoch_idx > 10:
                            #     self.save_com_spe_embed(config, epoch_idx, model_name = "Cleaner")
                        
                    else:
                        stop_flag += 1     
                self.logger.info('best result: \n'  + '\t' + dict2str(best_metric))
                self.logger.info('best epoch: \n'  + '\t' + dict2str(best_epoch,epoch=True) + '\n')

                if stop_flag > self.stopping_step:
                    stop_output = '+++++Finished training, best eval result in epoch %d' % \
                                  (epoch_idx - self.cur_step * self.eval_step)
                    if verbose:
                        self.logger.info(stop_output)
                    break
        return self.best_valid_score, self.best_valid_result, self.best_test_upon_valid

    

    @torch.no_grad()
    def evaluate(self, eval_data, is_test=False, idx=0):
        r"""Evaluate the model based on the eval data.
        Returns:
            dict: eval result, key is the eval metric and value in the corresponding metric value
        """
        self.model.eval()

        # batch full users
        batch_matrix_list = []
        for batch_idx, batched_data in enumerate(eval_data):
            # predict: interaction without item ids
            scores = self.model.full_sort_predict(batched_data)
            masked_items = batched_data[1]
            # mask out pos items
            scores[masked_items[0], masked_items[1]] = -1e10
            # rank and get top-k
            _, topk_index = torch.topk(scores, max(self.config['topk']), dim=-1)  # nusers x topk
            batch_matrix_list.append(topk_index)
        return self.evaluator.evaluate(batch_matrix_list, eval_data, is_test=is_test, idx=idx)

    def plot_train_loss(self, show=True, save_path=None):
        r"""Plot the train loss in each epoch

        Args:
            show (bool, optional): whether to show this figure, default: True
            save_path (str, optional): the data path to save the figure, default: None.
                                       If it's None, it will not be saved.
        """
        epochs = list(self.train_loss_dict.keys())
        epochs.sort()
        values = [float(self.train_loss_dict[epoch]) for epoch in epochs]
        plt.plot(epochs, values)
        plt.xticks(epochs)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        if show:
            plt.show()
        if save_path:
            plt.savefig(save_path)

