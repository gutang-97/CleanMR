# coding: utf-8


import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class AbstractRecommender(nn.Module):
    r"""Base class for all models
    """
    def pre_epoch_processing(self):
        pass

    def post_epoch_processing(self):
        pass

    def calculate_loss(self, interaction):
        r"""Calculate the training loss for a batch data.

        Args:
            interaction (Interaction): Interaction class of the batch.

        Returns:
            torch.Tensor: Training loss, shape: []
        """
        raise NotImplementedError

    def predict(self, interaction):
        r"""Predict the scores between users and items.

        Args:
            interaction (Interaction): Interaction class of the batch.

        Returns:
            torch.Tensor: Predicted scores for given users and items, shape: [batch_size]
        """
        raise NotImplementedError

    def full_sort_predict(self, interaction):
        r"""full sort prediction function.
        Given users, calculate the scores between users and all candidate items.

        Args:
            interaction (Interaction): Interaction class of the batch.

        Returns:
            torch.Tensor: Predicted scores for given users and all candidate items,
            shape: [n_batch_users * n_candidate_items]
        """
        raise NotImplementedError
    #
    # def __str__(self):
    #     """
    #     Model prints with number of trainable parameters
    #     """
    #     model_parameters = filter(lambda p: p.requires_grad, self.parameters())
    #     params = sum([np.prod(p.size()) for p in model_parameters])
    #     return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = self.parameters()
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

from sklearn.decomposition import PCA

class GeneralRecommender(AbstractRecommender):
    """This is a abstract general recommender. All the general model should implement this class.
    The base general recommender class provide the basic dataset and parameters information.
    """
    def __init__(self, config, dataloader):
        super(GeneralRecommender, self).__init__()

        # load dataset info
        self.USER_ID = config['USER_ID_FIELD']
        self.ITEM_ID = config['ITEM_ID_FIELD']
        self.NEG_ITEM_ID = config['NEG_PREFIX'] + self.ITEM_ID
        self.n_users = dataloader.dataset.get_user_num()               # 19445
        self.n_items = dataloader.dataset.get_item_num()               # 7050

        # load parameters info
        self.batch_size = config['train_batch_size']
        self.device = config['device']

        # load encoded features here
        self.v_feat, self.t_feat = None, None
        if not config['end2end'] and config['is_multimodal_model']:
            dataset_path = os.path.abspath(config['data_path'] + config['dataset'])
            # if file exist?
            v_feat_file_path = os.path.join(dataset_path, config['vision_feature_file'])
            t_feat_file_path = os.path.join(dataset_path, config['text_feature_file'])
            if os.path.isfile(v_feat_file_path):
                v_feat = np.load(v_feat_file_path, allow_pickle=True)
                V_PCA = PCA(n_components=384) #384
                v_feat = V_PCA.fit_transform(v_feat)
                self.v_feat = torch.from_numpy(v_feat).type(torch.FloatTensor).to(
                    self.device)               #(7050,4096)
                self.v_feat = torch.tanh(self.v_feat)
            if os.path.isfile(t_feat_file_path):
                t_feat = np.load(t_feat_file_path, allow_pickle=True)
                
                # T_PCA = PCA(n_components=256) 
                # t_feat = T_PCA.fit_transform(t_feat)
                
                self.t_feat = torch.from_numpy(t_feat).type(torch.FloatTensor).to(
                    self.device)           #(7050,384)
                
                # self.t_feat = F.normalize(self.t_feat)
                self.t_feat = torch.tanh(self.t_feat)
            assert self.v_feat is not None or self.t_feat is not None, 'Features all NONE'
