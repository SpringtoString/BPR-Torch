# -*- coding: utf-8 -*-
import pandas as pd
import torch
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import sys
import os

import torch.nn as nn
import numpy as np
import torch.utils.data as Data
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import log_loss, roc_auc_score
from collections import OrderedDict, namedtuple, defaultdict
import random
import multiprocessing
import heapq
import time
import sys

sys.path.append('../')
import util.metrics as metrics


data_generator = 0 # Data(path=filepath) 主程序中传入


def get_auc(item_score, user_pos_test):
    '''

    :param item_score: dict,待选item的预测评分
    :param user_pos_test: user 测试集中真实交互的item
    :return: auc
    '''
    item_score = sorted(item_score.items(), key=lambda kv: kv[1])
    item_score.reverse()

    item_sort = [x[0] for x in item_score]
    posterior = [x[1] for x in item_score]

    r = []
    for i in item_sort:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = metrics.auc(ground_truth=r, prediction=posterior)
    return auc

def test_one_user(x):
    # user u's ratings for user u
    rating = x[0]
    # uid
    u = x[1]
    ITEM_NUM = data_generator.n_items
    Ks = [20, 40, 60, 80, 100]
    # user u's items in the training set
    try:
        training_items = data_generator.train_items[u]  # user 已交互的item
    except Exception:
        training_items = []
    # user u's items in the test set
    user_pos_test = data_generator.test_set[u]  # 测试集中真实的item

    all_items = set(range(ITEM_NUM))

    test_items = list(all_items - set(training_items))  # 待选的item

    def ranklist_by_sorted(user_pos_test, test_items, rating, Ks):
        '''

        :param user_pos_test: user 测试集中真实交互的item
        :param test_items:    待选item
        :param rating:        user的所有预测评分
        :param Ks:            TOP-K
        :return:
        '''
        item_score = {}
        for i in test_items:
            item_score[i] = rating[i]

        K_max = max(Ks)
        K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)

        r = []
        for i in K_max_item_score:
            if i in user_pos_test:
                r.append(1)
            else:
                r.append(0)
        # auc = get_auc(item_score, user_pos_test)
        auc = 0
        return r, auc

    def get_performance(user_pos_test, r, auc, Ks):
        '''

        :param user_pos_test:    user 测试集中真实交互的item
        :param r:                r = [1,0,1] 表示预测TOP-K是否命中
        :param auc:              auc =0 标量
        :param Ks:               TOP-K
        :return:
        '''
        precision, recall, ndcg, hit_ratio, MAP = [], [], [], [], []

        for K in Ks:
            precision.append(metrics.precision_at_k(r, K))
            recall.append(metrics.recall_at_k(r, K, len(user_pos_test)))
            ndcg.append(metrics.ndcg_at_k(r, K))
            hit_ratio.append(metrics.hit_at_k(r, K))
            MAP.append(metrics.AP_at_k(r, K, len(user_pos_test)))

        return {'recall': np.array(recall), 'precision': np.array(precision), 'ndcg': np.array(ndcg),
                'hit_ratio': np.array(hit_ratio), 'MAP': np.array(MAP), 'auc': auc}

    # if args.test_flag == 'part':
    #     r, auc = ranklist_by_heapq(user_pos_test, test_items, rating, Ks)
    # else:

    r, auc = ranklist_by_sorted(user_pos_test, test_items, rating, Ks)

    return get_performance(user_pos_test, r, auc, Ks)


class BPR(nn.Module):

    def __init__(self, n_user, n_item,
                 embedding_size=4, l2_reg_linear=0.00001, l2_reg_embedding=0.00001, l2_reg_dnn=0, init_std=0.0001,
                 seed=1024,
                 device='cpu'):

        super(BPR, self).__init__()

        self.n_user = n_user
        self.n_item = n_item
        self.embedding_size = embedding_size
        self.device = device
        self.l2_reg_embedding = l2_reg_embedding
        self.embedding_dict = nn.ModuleDict({
            'user_emb': self.create_embedding_matrix(n_user, embedding_size),
            'item_emb': self.create_embedding_matrix(n_item, embedding_size)
        })

        self.to(device)

    def forward(self, input_dict):
        '''

        :param input_dict:
        :return:   rui, ruj
        '''
        users, pos_items, neg_items = input_dict['users'], input_dict['pos_items'], input_dict['neg_items']

        user_vector = self.embedding_dict['user_emb'](users)
        pos_items_vector = self.embedding_dict['item_emb'](pos_items)
        neg_items_vector = self.embedding_dict['item_emb'](neg_items)

        rui = torch.sum(torch.mul(user_vector, pos_items_vector), dim=-1, keepdim=True)
        ruj = torch.sum(torch.mul(user_vector, neg_items_vector), dim=-1, keepdim=True)

        emb_loss = torch.norm(user_vector) ** 2 + torch.norm(pos_items_vector) ** 2 + torch.norm(neg_items_vector) ** 2

        return rui, ruj, emb_loss

    def rating(self, user_batch, all_item):
        user_vector = self.embedding_dict['user_emb'](user_batch)
        pos_items_vector = self.embedding_dict['item_emb'](all_item)
        return torch.mm(user_vector, pos_items_vector.t())

    def fit(self, learning_rate=0.001, batch_size=500, epochs=15, verbose=5, early_stop=False):

        print(self.device, end="\n")
        self.data_generator = data_generator
        model = self.train()
        loss_func = nn.LogSigmoid()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0)
        # 显示 一次epoch需要几个step
        sample_num = data_generator.n_train
        n_batch = (sample_num - 1) // batch_size + 1

        print("Train on {0} samples,  {1} steps per epoch".format(sample_num, n_batch))

        logloss_list = []
        auc_score_list = []

        for epoch in range(epochs):
            loss_epoch = 0
            total_loss, total_mf_loss, total_emb_loss = 0.0, 0.0, 0.0
            train_result = {}
            pred_ans = []
            true_ans = []
            with torch.autograd.set_detect_anomaly(True):
                start_time = time.time()
                for index in range(n_batch):
                    users, pos_items, neg_items = data_generator.sample(batch_size)

                    users = torch.from_numpy(np.array(users)).to(self.device).long()
                    pos_items = torch.from_numpy(np.array(pos_items)).to(self.device).long()
                    neg_items = torch.from_numpy(np.array(neg_items)).to(self.device).long()

                    input_dict = {'users': users, 'pos_items': pos_items, 'neg_items': neg_items}
                    rui, ruj, emb_loss = model(input_dict)

                    optimizer.zero_grad()

                    mf_loss = -loss_func(rui - ruj).mean()
                    reg_emb_loss = self.l2_reg_embedding*emb_loss/batch_size
                    loss = mf_loss + reg_emb_loss

                    loss.backward(retain_graph=True)
                    optimizer.step()

                    total_mf_loss = total_mf_loss + mf_loss.item()
                    total_emb_loss = total_emb_loss + reg_emb_loss.item()
                    total_loss = total_loss + loss.item()

                if verbose > 0:
                    epoch_time = time.time() - start_time
                    print('epoch %d %.2fs train loss is [%.4f = %.4f + %.4f] ' % (epoch, epoch_time,
                                total_loss / n_batch, total_mf_loss/n_batch, total_emb_loss/n_batch))

            if verbose>0 and epoch%verbose==0:
                start_time = time.time()
                result = self.test(batch_size=batch_size)
                eval_time = time.time() - start_time

                print(
                    'epoch %d %.2fs test precision is [%.4f %.4f] recall is [%.4f %.4f] ndcg is [%.4f %.4f] hit_ratio is [%.4f %.4f] MAP is [%.4f %.4f] auc is %.4f ' %
                    (epoch, eval_time,
                     result['precision'][0], result['precision'][-1],
                     result['recall'][0], result['recall'][-1],
                     result['ndcg'][0],result['ndcg'][-1],
                     result['hit_ratio'][0],result['hit_ratio'][-1],
                     result['MAP'][0], result['MAP'][-1],
                     result['auc']))

            print(" ")

    def test(self, batch_size=256, ):
        model = self.eval()
        cores = multiprocessing.cpu_count() // 2
        pool = multiprocessing.Pool(cores)

        Ks = [20, 40, 60, 80, 100]
        # data_generator = self.data_generator
        ITEM_NUM = data_generator.n_items
        result = {'precision': np.zeros(len(Ks)), 'recall': np.zeros(len(Ks)), 'ndcg': np.zeros(len(Ks)),
                  'hit_ratio': np.zeros(len(Ks)), 'MAP': np.zeros(len(Ks)), 'auc': 0.}

        u_batch_size = batch_size
        i_batch_size = batch_size

        test_users = list(data_generator.test_set.keys())
        n_test_users = len(test_users)
        n_user_batchs = (n_test_users - 1) // u_batch_size + 1

        count = 0
        # with torch.no_grad():
        for u_batch_id in range(n_user_batchs):
            start = u_batch_id * u_batch_size
            # end 这里需要完善
            end = (u_batch_id + 1) * u_batch_size
            user_batch = test_users[start: end]  # 取一部分 user

            all_item = range(ITEM_NUM)

            user_batch = torch.from_numpy(np.array(user_batch)).to(self.device).long()
            all_item = torch.from_numpy(np.array(all_item)).to(self.device).long()
            rate_batch = model.rating(user_batch,
                                      all_item).detach().cpu()  # shape is [len(user_batch),ITEM_NUM] 即预测评分矩阵

            user_batch_rating_uid = zip(rate_batch.numpy(), user_batch.detach().cpu().numpy())  # 一个user 对应一行评分
            batch_result = pool.map(test_one_user, user_batch_rating_uid)
            count += len(batch_result)

            for re in batch_result:
                result['precision'] += re['precision'] / n_test_users
                result['recall'] += re['recall'] / n_test_users
                result['ndcg'] += re['ndcg'] / n_test_users
                result['hit_ratio'] += re['hit_ratio'] / n_test_users
                result['MAP'] += re['MAP'] / n_test_users
                # result['auc'] += re['auc'] / n_test_users

        assert count == n_test_users
        pool.close()
        return result

    def create_embedding_matrix(self, vocabulary_size, embedding_size, init_std=0.0001, sparse=False,):
        embedding = nn.Embedding(vocabulary_size, embedding_size, sparse=sparse)
        nn.init.normal_(embedding.weight, mean=0, std=init_std)

        return embedding



