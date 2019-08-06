
# with open("user_lists") as f:
#     data = eval(f.read())
#     print(data)
# import collections
# dic = collections.defaultdict(set)
# dic[1].add(2)
# dic = dict(dic)
# print(dic)
import random
import numpy as np
import torch
import utils
import collections
import copy
from parse import arg_parse
from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score
from utils import get_test_batch_ctr, sample_test_item
import time
import json
import math
def ctr_evaluate(model, train_user_lists, train_item_lists, test_user_lists, neg_lists, k_shot, args, output_dim):
    '''
    model evaluate metrics: logloss and AUC
    '''
    # preprocess the adj matrix to k shot
    # dics = json.dumps(test_user_lists)
    # f = open("1.json", "w")
    # f.write(dics)
    # f.close()
    adj_lists_temp = train_user_lists.copy()
    adj_lists_temp.update(test_user_lists)
    adj_lists_temp_n_k = collections.defaultdict(list)
    item_lists_temp = train_item_lists.copy()
    # add test items to item_lists_temp
    for user, items in test_user_lists.items():
        for item in items:
            mytemp = copy.copy(item_lists_temp[item])
            mytemp.append(user)
            item_lists_temp[item] = mytemp

    # k shot for test
    for user in list(test_user_lists.keys()):
        pos_items = list(np.random.choice(test_user_lists[user], k_shot,replace=False))
        # remove items in item_lists
        remove_items = set(test_user_lists[user])-set(pos_items)
        for remove_item in remove_items:
            adj_lists_temp_n_k[user].append(remove_item)
            temp = copy.copy(item_lists_temp[remove_item])
            temp.remove(user)
            item_lists_temp[remove_item] = temp
            adj_lists_temp_n_k[remove_item].append(user)
        adj_lists_temp[user] = pos_items

    adj_lists_temp.update(item_lists_temp)
    # so far, adj_lists_temp and item_lists_temp serves as finetune data
    print("Finetune data prepared!")
    # time3 = time.time()
    logloss_list = []
    roc_auc_list = []
    for user in list(test_user_lists.keys()):
        model_tmp = copy.deepcopy(model)
        model_tmp.train()
        optimizer_tmp = torch.optim.Adam(model_tmp.parameters(),lr=args.inner_lr)
        for inner_batch in range(args.inner_batch):
            # adj_lists_sample.update(user_lists_test)
            # pos_items,neg_items,adj_lists_sample = get_train_batch(users,train_user_lists,train_item_lists,k_shot,num_users,num_items)
            items,labels = get_test_batch_ctr([user],adj_lists_temp,k_shot,neg_lists)
            labels = torch.tensor(labels).cuda()
            optimizer_tmp.zero_grad()
            
            # loss = model_tmp.loss(users,pos_items,neg_items,adj_lists_sample,few_shot=k_shot)
            loss = model_tmp.loss_ctr([user],items,labels,adj_lists_temp, few_shot=k_shot)
            loss.backward()
            optimizer_tmp.step()
        model_tmp.eval()
        items, n_k,labels = sample_test_item(user,adj_lists_temp_n_k,neg_lists,k_shot)
        num_items = len(items)
        user_embedding = model_tmp.forward([user], adj_lists_temp)
        user_embedding = torch.unsqueeze(user_embedding, 1).repeat([1, num_items, 1])
        items = np.reshape(items, -1)
        item_embedding = model_tmp.forward(items, adj_lists_temp)
        item_embedding = item_embedding.view([-1, num_items, output_dim])
        pred = torch.sigmoid(torch.sum(item_embedding * user_embedding, -1).view(-1)) 
        labels = labels
        labels = torch.tensor(labels).cuda()
        labels = labels.view(-1)
        test_logloss = log_loss(labels.cpu().detach().numpy(), pred.cpu().detach().numpy())
        logloss_list.append(test_logloss)
        test_roc_auc = roc_auc_score(labels.cpu().detach().numpy(), pred.cpu().detach().numpy())
        roc_auc_list.append(test_roc_auc)
        # print("Logloss: " , logloss_list, "ROC AUC Score: ", roc_auc_list)
    # time6 = time.time()
    # print(time6-time3)
    logloss = np.mean(logloss_list)
    auc = np.mean(roc_auc_list)
    print("Logloss: ", logloss, "ROC AUC Score: ", auc)
    # dics = json.dumps(test_user_lists)
    # f = open("2.json", "w")
    # f.write(dics)
    # f.close()
    return logloss,auc


def ctr_evaluate_gnn(model, train_user_lists, train_item_lists, test_user_lists, neg_lists, k_shot, args, output_dim):
    '''
    model evaluate metrics: logloss and AUC
    '''
    # preprocess the adj matrix to k shot
    adj_lists_temp = train_user_lists.copy()
    adj_lists_temp_n_k = collections.defaultdict(list)
    item_lists_temp = train_item_lists.copy()
    # add test items to item_lists_temp
    for user, items in test_user_lists.items():
        for item in items:
            mytemp = copy.copy(item_lists_temp[item])
            mytemp.append(user)
            item_lists_temp[item] = mytemp

    # k shot for test
    for user in list(test_user_lists.keys()):
        pos_items = list(np.random.choice(test_user_lists[user], k_shot,replace=False))
        # remove items in item_lists
        remove_items = set(test_user_lists[user])-set(pos_items)
        for remove_item in remove_items:
            adj_lists_temp_n_k[user].append(remove_item)
            temp = copy.copy(item_lists_temp[remove_item])
            temp.remove(user)
            item_lists_temp[remove_item] = temp
            adj_lists_temp_n_k[remove_item].append(user)
        test_user_lists[user] = pos_items
    adj_lists_temp.update(test_user_lists)
    adj_lists_temp.update(item_lists_temp)
    # so far, adj_lists_temp and item_lists_temp serves as finetune data
    print("Finetune data prepared!")
    # time3 = time.time()
    logloss_list = []
    roc_auc_list = []
    for user in list(test_user_lists.keys()):
        model_tmp = copy.deepcopy(model)
        model_tmp.train()
        optimizer_tmp = torch.optim.Adam(model_tmp.parameters(),lr=args.inner_lr)
        for inner_batch in range(args.inner_batch):
            # adj_lists_sample.update(user_lists_test)
            # pos_items,neg_items,adj_lists_sample = get_train_batch(users,train_user_lists,train_item_lists,k_shot,num_users,num_items)
            items,labels = get_test_batch_ctr([user],adj_lists_temp,k_shot,neg_lists)
            labels = torch.tensor(labels).cuda()
            optimizer_tmp.zero_grad()
            
            # loss = model_tmp.loss(users,pos_items,neg_items,adj_lists_sample,few_shot=k_shot)
            loss = model_tmp.loss_ctr([user],items,labels,adj_lists_temp, few_shot=k_shot)
            loss.backward()
            optimizer_tmp.step()
        model_tmp.eval()
        items, n_k,labels = sample_test_item(user,adj_lists_temp_n_k,neg_lists,k_shot)
        num_items = len(items)
        user_embedding = model_tmp.forward([user], adj_lists_temp)
        user_embedding = torch.unsqueeze(user_embedding, 1).repeat([1, num_items, 1])
        items = np.reshape(items, -1)
        item_embedding = model_tmp.forward(items, adj_lists_temp)
        item_embedding = item_embedding.view([-1, num_items, output_dim])
        pred = torch.sum(item_embedding * user_embedding, -1).view(-1)
        labels = labels
        labels = torch.tensor(labels).cuda()
        labels = labels.view(-1)
        test_logloss = log_loss(labels.cpu().detach().numpy(), pred.cpu().detach().numpy())
        logloss_list.append(test_logloss)
        test_roc_auc = roc_auc_score(labels.cpu().detach().numpy(), pred.cpu().detach().numpy())
        roc_auc_list.append(test_roc_auc)
        # print("Logloss: " , logloss_list, "ROC AUC Score: ", roc_auc_list)
    # time6 = time.time()
    # print(time6-time3)
    logloss = np.mean(logloss_list)
    auc = np.mean(roc_auc_list)
    print("Logloss: ", logloss, "ROC AUC Score: ", auc)
    return logloss,auc