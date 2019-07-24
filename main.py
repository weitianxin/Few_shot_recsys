import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import random
import seaborn as sbs
from torch.autograd import Variable
from tqdm import tqdm_notebook as tqdm
import copy
from parse import arg_parse
from model import graphsage
from model_gcn import gcn
import collections
import time
import scipy.sparse as sp

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

import os
def gowalla(ratio=0.8):
    data_dir = './data/gowalla/'
    train_dir = data_dir + "train.txt"
    test_dir = data_dir + "test.txt"
    user_lists = collections.defaultdict(list)
    train_item_lists = collections.defaultdict(list)
    n_items = 40981
    n_users = 29858
    R = sp.dok_matrix((n_users+n_items, n_users+n_items), dtype=np.float32)
    with open(train_dir) as f:
        for l in f.readlines():
            if len(l) > 0:
                l = l.strip('\n').split(' ')
                items = [29858+int(i) for i in l[1:]]
                uid = int(l[0])
                user_lists[uid].extend(items)
                for item in items:
                    R[uid,item] = 1
                    R[item, uid] = 1


    with open(test_dir) as f:
        for l in f.readlines():
            if len(l) > 0:
                l = l.strip('\n').split(' ')
                items = [29858+int(i) for i in l[1:]]
                uid = int(l[0])
                user_lists[uid].extend(items)
                for item in items:
                    R[uid,item] = 1
                    R[item, uid] = 1

    users = list(user_lists.keys())
    random.shuffle(users)
    random.seed(2019)
    bound = int(ratio*n_users)
    train_index = users[:bound]
    test_index = users[bound:]
    train_user_lists = {i:user_lists[i] for i in train_index}
    test_user_lists = {i:user_lists[i] for i in test_index}
    #build item lists
    for user,items in train_user_lists.items():
        for item in items:
            train_item_lists[item].append(user)
    # coo = R.tocoo().astype(np.float32)
    # indices = np.mat([coo.row, coo.col])
    # R = torch.sparse.FloatTensor(torch.LongTensor(indices), torch.FloatTensor(coo.data), coo.shape).cuda()
    # data = torch.ones([n_items+n_users,32]).cuda()
    # time1 = time.time()
    # for i in range(3):
    #     torch.sparse.mm(R,data)
    # time2 = time.time()
    # print(time2-time1)
    return R,train_user_lists,test_user_lists,train_item_lists,n_users,n_items

def get_train_batch(users,user_lists,item_lists,k_shot, n_users, n_items):
    adj_lists_sample = copy.deepcopy(user_lists)
    item_lists_sample = copy.deepcopy(item_lists)
    pos_list,neg_list = [],[]
    for user in users:
        pos_items = list(np.random.choice(user_lists[user], k_shot))
        # remove items in item_lists
        remove_items = set(user_lists[user])-set(pos_items)
        for remove_item in remove_items:
            item_lists_sample[remove_item].remove(user)
        adj_lists_sample[user] = pos_items
        neg_items = []
        while len(neg_items)!=k_shot:
            neg_item = np.random.randint(n_users,n_users+n_items)
            if neg_item not in pos_items:
                neg_items.append(neg_item)
        pos_list.append(pos_items)
        neg_list.append(neg_items)

    adj_lists_sample.update(item_lists_sample)

    return pos_list,neg_list,adj_lists_sample

def replace_grad(parameter_gradients, parameter_name):
    """Creates a backward hook function that replaces the calculated gradient
    with a precomputed value when .backward() is called.

    See
    https://pytorch.org/docs/stable/autograd.html?highlight=hook#torch.Tensor.register_hook
    for more info
    """
    def replace_grad_(module):
        return parameter_gradients[parameter_name]

    return replace_grad_

def evaluate():
    pass

def build_subgraph(nodes,adj_lists,R,k_hop):
    time1 = time.time()
    nodes_list = [nodes]
    for i in range(k_hop):
        temp_nodes = []
        print(len(nodes_list[i]))
        for node in nodes_list[i]:
            temp = adj_lists[node]
            temp+=[node]
            temp_nodes+=temp
        neighbors = list(set(temp_nodes))
        nodes_list.append(neighbors)
    time2 = time.time()
    print(time2 - time1)

    unique_nodes_list = nodes_list[-1]
    # print(len(unique_nodes_list),len(set(unique_nodes_list)))
    # unique_nodes = {n: i for i, n in enumerate(unique_nodes_list)}
    # adj = np.zeros((len(unique_nodes), len(unique_nodes)))
    # for node in nodes_list[-2]:
    #     c = unique_nodes
    #     for n in adj_lists[node]:[node]
    #         n = unique_nodes[n]
    #         adj[c,n] = 1
    #         adj[n,c] = 1
    R_sample = R[unique_nodes_list]
    print(R_sample.shape)
    time3 =  time.time()
    print(time3-time2)
    exit()

    # column_indices = [unique_nodes[n] for node in nodes_list[-2] for n in adj_lists[node]]
    # row_indices = [i for i in range(len(unique_nodes)) for j in range(len(unique_nodes))]
    # adj[row_indices, column_indices] = 1
    return adj

args = arg_parse()
#device
os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
R,train_user_lists,test_user_lists,train_item_lists,num_users,num_items  = gowalla()
print(num_items,num_users)
train_user_lists
#model
meta_train_loss = []
meta_test_loss = []
lr_inner = 0.01
#0-1正态分布
feat_data = nn.Embedding(num_users+num_items,embedding_dim=args.embed_dim).cuda()
feat_data.weight.requires_grad = False
model = graphsage(feat_data,args.input_dim,args.hidden_dim,args.output_dim,args.n_layer).cuda()
# few_shot_recsys
#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=args.lr_decay, patience = args.patience, mode='max', threshold=args.threshold)
optimizer = torch.optim.Adam(model.parameters(),lr=1e-4,weight_decay=1e-4)
for epoch in np.arange(args.n_epoch) + 1:
    for meta_batch in np.arange(args.n_batch):
        task_grads = []
        task_losss = []
        '''
        Cumulate Inner Gradient
        '''
        users = np.random.choice(list(train_user_lists.keys()), args.batch_size, replace=False)
        for user in users:
            model_tmp = copy.deepcopy(model)
            model_tmp.train()
            optimizer_tmp = torch.optim.Adam(model_tmp.parameters(),lr=1e-4)
            time1 = time.time()
            for inner_batch in range(args.inner_batch):
                k_shot = np.random.randint(args.n_shot) + 1
                pos_items,neg_items,adj_lists_sample = get_train_batch([user],train_user_lists,train_item_lists,k_shot,num_users,num_items)
                optimizer_tmp.zero_grad()
                #---change
                # users,pos_items,neg_items = torch.tensor(users).to(device),torch.tensor(pos_items).to(device),torch.tensor(neg_items).to(device)
                # temp = [user]
                # temp.extend(pos_items[0])
                # temp.extend(neg_items[0])
                build_subgraph(temp, adj_lists_sample, R, args.n_layer)

                loss = model_tmp.loss([user],pos_items,neg_items,adj_lists_sample,few_shot=k_shot)
                # x, y = task.train_x.to(device), Variable(task.train_y).to(device)
                # pred_y = model_tmp.forward(x)
                # loss = criterion(pred_y, y)
                #根据loss backward获得各可训练参数的梯度
                loss.backward()
                #optimizer的step结合梯度信息和更新算法来实际改变参数的值
                optimizer_tmp.step()
            time2 = time.time()
            model_tmp.eval()
            optimizer_tmp.zero_grad()
            k_shot = np.random.randint(args.n_shot) + 1
            pos_items, neg_items, adj_lists_sample = get_train_batch([user],train_user_lists,train_item_lists
                                                                            ,k_shot,num_users,num_items)
            #todo
            loss = model_tmp.loss([user],pos_items,neg_items,adj_lists_sample,few_shot=k_shot)

            task_losss += [loss.cpu().detach().numpy()]
            loss.backward()
            task_grads += [{name: param.grad for (name, param) in model_tmp.named_parameters()}]
        meta_train_loss += [np.average(task_losss)]
        print(meta_train_loss)

        '''
        Meta-Update
        '''
        avg_task_grad = {name: torch.stack([name_grad[name] for name_grad in task_grads]).mean(dim=0)
                         for name in task_grads[0].keys()}

        hooks = []
        for name, param in model.named_parameters():
            hooks.append(
                param.register_hook(replace_grad(avg_task_grad, name))
            )

        loss = model.loss(users,pos_items,neg_items,adj_lists_sample)
        loss.backward()
        optimizer.step()

        for h in hooks:
            h.remove()
    plt.plot(meta_train_loss, label='meta_train')
    # plt.plot(meta_test_loss, label='meta_test')
    plt.legend()
    plt.show()
    # plt.plot(target_task.test_x.cpu().detach().numpy(), pred_y.cpu().detach().numpy(), '^--', label='predict')
    # target_task.plot()
    # plt.legend()
    # plt.show()
    '''
    Evaluate:
    '''
    # model_tmp = copy.deepcopy(model)
    # model_tmp.train()
    # optimizer_tmp = torch.optim.Adam(model_tmp.parameters())
    # for inner_batch in range(4):
    #     optimizer_tmp.zero_grad()
    #     x, y = target_task.train_x.to(device), Variable(target_task.train_y).to(device)
    #     pred_y = model_tmp.forward(x)
    #     loss = criterion(pred_y, y)
    #     loss.backward()
    #     optimizer_tmp.step()
    # model_tmp.eval()
    # x, y = target_task.test_x.to(device), Variable(target_task.test_y).to(device)
    # pred_target = model_tmp.forward(x)
    # loss = criterion(pred_target, y)
    # meta_test_loss += [loss.cpu().detach().numpy()]
