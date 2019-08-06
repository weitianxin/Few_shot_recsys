import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import random
from torch.autograd import Variable
from tqdm import tqdm_notebook as tqdm
import copy
from parse import arg_parse
from model import graphsage
from model_gcn import gcn
import collections
import time
import scipy.sparse as sp
from load_data import *
from utils import get_train_batch,get_train_batch_ctr
import os
from test import ctr_evaluate,ctr_evaluate_gnn
import json
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

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

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

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
    # column_indices = [unique_nodes[n] for node in nodes_list[-2] for n in adj_lists[node]]
    # row_indices = [i for i in range(len(unique_nodes)) for j in range(len(unique_nodes))]
    # adj[row_indices, column_indices] = 1

def gnn(model,optimizer,scheduler,neg_lists,train_user_lists, val_user_lists, test_user_lists, train_item_lists, num_users, num_items):
    best_auc = 0
    val_less_time = 0
    for epoch in np.arange(args.n_epoch) + 1:
        for batch in np.arange(args.n_batch):
            #每个batch里有batch_size个user
            for i in range(args.batch_size):
                users = np.random.choice(list(train_user_lists.keys()), 1, replace=False)
                k_shot = np.random.randint(args.n_shot) + 1
                # pos_items,neg_items,adj_lists_sample = get_train_batch(users,train_user_lists,train_item_lists,k_shot,num_users,num_items)
                items,labels,adj_lists_sample = get_train_batch_ctr(users,train_user_lists,train_item_lists,k_shot,neg_lists)
                labels = torch.tensor(labels).cuda()
                optimizer.zero_grad()
                # loss = model_tmp.loss(users,pos_items,neg_items,adj_lists_sample,few_shot=k_shot)
                loss = model.loss_ctr(users,items,labels,adj_lists_sample, few_shot=k_shot)
                #根据loss backward获得各可训练参数的梯度
                loss.backward()
                #optimizer的step结合梯度信息和更新算法来实际改变参数的值
                optimizer.step()

        logloss,auc = ctr_evaluate_gnn(model, train_user_lists, train_item_lists, val_user_lists, neg_lists, args.n_shot, args, args.output_dim)
        print(epoch," auc:",str(auc))
        with open(args.test_val_file,"a+") as f:
            f.write(str(auc)+"\n")
        #early stop
        if optimizer.param_groups[0]['lr'] < args.lr_early_stop:
            logloss,auc = ctr_evaluate_gnn(model, train_user_lists, train_item_lists, test_user_lists, neg_lists, args.n_shot, args, args.output_dim)
            print("finish training")
            with open(args.test_val_file,"a+") as f:
                f.write("test auc:"+str(auc)+" best val auc:"+str(best_auc))
            torch.save(model.state_dict(), "./checkpoint/best_{}.pt".format(args.model))
            break
        #adjust lr
        scheduler.step(auc)
        #best performance in val
        if auc>=best_auc:
            best_auc = auc

def gnn_few_shot(model,optimizer,scheduler,neg_lists,train_user_lists, val_user_lists, test_user_lists, train_item_lists, num_users, num_items):
    pass

def maml():
    pass

if __name__ == "__main__":
    args = arg_parse()
    #device
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    # R,train_user_lists,test_user_lists,train_item_lists,num_users,num_items  = gowalla()
    neg_lists,train_user_lists, val_user_lists, test_user_lists, train_item_lists, num_users, num_items  = gowalla()
    print(num_items,num_users)
    #model
    meta_train_loss = []
    meta_test_loss = []
    #0-1正态分布
    feat_data = nn.Embedding(num_users+num_items,embedding_dim=args.embed_dim).cuda()
    feat_data.weight.requires_grad = False
    model = graphsage(feat_data,args.input_dim,args.hidden_dim,args.output_dim,args.n_layer).cuda()
    # few_shot_recsys
    
    optimizer = torch.optim.Adam(model.parameters(),lr=args.lr)#,weight_decay=1e-4)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=args.lr_decay, patience = args.patience, mode='max', threshold=args.threshold)

    # if args.model=="gnn":
    #     gnn(model,optimizer,scheduler,neg_lists,train_user_lists, val_user_lists, test_user_lists, train_item_lists, num_users, num_items )
    # elif args.model=="gnn_few_shot":
    #     gnn_few_shot(model,optimizer,scheduler,neg_lists,train_user_lists, val_user_lists, test_user_lists, train_item_lists, num_users, num_items)
    # elif args.model=="maml":
    best_auc = 0
    val_less_time = 0
    for epoch in np.arange(args.n_epoch) + 1:
        for meta_batch in np.arange(args.n_batch):
            #每个meta_batch里有batch_size个task
            task_grads = []
            task_losss = []
            for i in range(args.batch_size):
                '''
                Cumulate Inner Gradient
                '''
                users = np.random.choice(list(train_user_lists.keys()), args.task_size, replace=False)
                model_tmp = copy.deepcopy(model)
                model_tmp.train()
                optimizer_tmp = torch.optim.Adam(model_tmp.parameters(),lr=args.inner_lr)
                for inner_batch in range(args.inner_batch):
                    k_shot = np.random.randint(args.n_shot) + 1
                    pos_items,neg_items,adj_lists_sample = get_train_batch(users,train_user_lists,train_item_lists,k_shot,num_users,num_items)
                    items,labels,adj_lists_sample = get_train_batch_ctr(users,train_user_lists,train_item_lists,k_shot,neg_lists)
                    labels = torch.tensor(labels).cuda()
                    optimizer_tmp.zero_grad()
                    # loss = model_tmp.loss(users,pos_items,neg_items,adj_lists_sample,few_shot=k_shot)
                    loss = model_tmp.loss_ctr(users,items,labels,adj_lists_sample, few_shot=k_shot)
                    # 根据loss backward获得各可训练参数的梯度
                    loss.backward()
                    #optimizer的step结合梯度信息和更新算法来实际改变参数的值
                    optimizer_tmp.step()
                #generarlized loss
                model_tmp.eval()
                optimizer_tmp.zero_grad()
                k_shot = np.random.randint(args.n_shot) + 1
                # pos_items,neg_items,adj_lists_sample = get_train_batch(users,train_user_lists,train_item_lists,k_shot,num_users,num_items)
                items,labels,adj_lists_sample = get_train_batch_ctr(users,train_user_lists,train_item_lists,k_shot,neg_lists)
                labels = torch.tensor(labels).cuda()
                
                loss = model_tmp.loss_ctr(users,items,labels,adj_lists_sample,few_shot=k_shot)
                # loss = model_tmp.loss(users,pos_items,neg_items,adj_lists_sample,few_shot=k_shot)
                task_losss += [loss.cpu().detach().numpy()]
                loss.backward()
                task_grads += [{name: param.grad for (name, param) in model_tmp.named_parameters() if param.requires_grad}]
                # print(task_grads[0])
            meta_train_loss += [np.average(task_losss)]
            with open("train_lr{}_innerlr{}.txt".format(args.lr,args.inner_lr),"a+") as f:
                f.write(str(meta_train_loss[-1])+"\n")
            '''
            Meta-Update
            '''
            avg_task_grad = {name: torch.stack([name_grad[name] for name_grad in task_grads]).mean(dim=0)
                             for name in task_grads[0].keys()}

            hooks = []
            for name, param in model.named_parameters():
                if param.requires_grad:
                    hooks.append(
                        param.register_hook(replace_grad(avg_task_grad, name))
                    )

            loss = model.loss_ctr(users,items,labels,adj_lists_sample,few_shot=k_shot)
            # loss = model.loss(users,pos_items,neg_items,adj_lists_sample,few_shot=k_shot)
            loss.backward()
            optimizer.step()

            for h in hooks:
                h.remove()
        print(epoch,":",meta_train_loss[-1])
        logloss,auc = ctr_evaluate(model, train_user_lists, train_item_lists, val_user_lists, neg_lists, args.n_shot, args, args.output_dim)
        print(epoch," auc:",str(auc))
        with open(args.test_val_file,"a+") as f:
            f.write(str(auc)+"\n")
        #early stop
        if optimizer.param_groups[0]['lr'] < args.lr_early_stop:
            logloss,auc = ctr_evaluate(model, train_user_lists, train_item_lists, test_user_lists, neg_lists, args.n_shot, args, args.output_dim)
            print("finish training")
            with open(args.test_val_file,"a+") as f:
                f.write("test auc:"+str(auc)+" best val auc:"+str(best_auc))
            torch.save(model.state_dict(), "./checkpoint/best_{}.pt".format(args.model))
            break
        #adjust lr
        # scheduler.step(auc)
        #best performance in val
        if auc>=best_auc:
            best_auc = auc
        if val_less_time>args.early_stop_epoches:
            logloss,auc = ctr_evaluate(model, train_user_lists, train_item_lists, test_user_lists, neg_lists, args.n_shot, args, args.output_dim)
            print("test auc:",auc)
            torch.save(model.state_dict(), "./checkpoint/final.pt")
    #     FOR TEST
    # args = arg_parse()
    # #device
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    # # R,train_user_lists,test_user_lists,train_item_lists,num_users,num_items  = gowalla()
    # neg_lists,train_user_lists, val_user_lists, test_user_lists, train_item_lists, num_users, num_items = gowalla_test()
    # print(num_items,num_users)
    # #model
    # meta_train_loss = []
    # meta_test_loss = []
    # #0-1正态分布
    # feat_data = nn.Embedding(num_users+num_items,embedding_dim=args.embed_dim).cuda()
    # feat_data.weight.requires_grad = False
    # model = graphsage(feat_data,args.input_dim,args.hidden_dim,args.output_dim,args.n_layer).cuda()
    # # few_shot_recsys
    # #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=args.lr_decay, patience = args.patience, mode='max', threshold=args.threshold)
    # optimizer = torch.optim.Adam(model.parameters(),lr=args.lr)#,weight_decay=1e-4)
    # for epoch in np.arange(args.n_epoch) + 1:
    #     for meta_batch in np.arange(args.n_batch):
    #         #每个meta_batch里有batch_size个task
    #         task_grads = []
    #         task_losss = []
    #         for i in range(args.batch_size):

    #             '''
    #             Cumulate Inner Gradient
    #             '''
    #             users = np.random.choice(list(train_user_lists.keys()), args.task_size, replace=False)
    #             model_tmp = copy.deepcopy(model)
    #             model_tmp.train()
    #             optimizer_tmp = torch.optim.Adam(model_tmp.parameters(),lr=args.inner_lr)
    #             for inner_batch in range(args.inner_batch):
    #                 k_shot = np.random.randint(args.n_shot) + 1
    #                 pos_items,neg_items,adj_lists_sample = get_train_batch(users,train_user_lists,train_item_lists,k_shot,num_users,num_items)
    #                 optimizer_tmp.zero_grad()
    #                 loss = model_tmp.loss(users,pos_items,neg_items,adj_lists_sample,few_shot=k_shot)
    #                 #根据loss backward获得各可训练参数的梯度
    #                 loss.backward()
    #                 #optimizer的step结合梯度信息和更新算法来实际改变参数的值
    #                 optimizer_tmp.step()
    #             #generarlized loss
    #             model_tmp.eval()
    #             optimizer_tmp.zero_grad()
    #             k_shot = np.random.randint(args.n_shot) + 1
    #             pos_items, neg_items, adj_lists_sample = get_train_batch(users,train_user_lists,train_item_lists
    #                                                                             ,k_shot,num_users,num_items)
    #             #todo
    #             loss = model_tmp.loss(users,pos_items,neg_items,adj_lists_sample,few_shot=k_shot)

    #             task_losss += [loss.cpu().detach().numpy()]
    #             loss.backward()
    #             task_grads += [{name: param.grad for (name, param) in model_tmp.named_parameters() if param.requires_grad}]
    #         meta_train_loss += [np.average(task_losss)]
    #         print(meta_train_loss[-1])

    #         '''
    #         Meta-Update
    #         '''
    #         avg_task_grad = {name: torch.stack([name_grad[name] for name_grad in task_grads]).mean(dim=0)
    #                          for name in task_grads[0].keys()}

    #         hooks = []
    #         for name, param in model.named_parameters():
    #             if param.requires_grad:
    #                 hooks.append(
    #                     param.register_hook(replace_grad(avg_task_grad, name))
    #                 )

    #         loss = model.loss(users,pos_items,neg_items,adj_lists_sample,few_shot=k_shot)
    #         loss.backward()
    #         optimizer.step()

    #         for h in hooks:
    #             h.remove()
    #     print(epoch,":",meta_train_loss[-1])
    #     logloss,auc = ctr_evaluate(model, train_user_lists, train_item_lists, val_user_lists, neg_lists, args.n_shot, args, args.output_dim)
    #     print(epoch," auc:",str(auc))
    #     with open(args.test_val_file,"a+") as f:
    #         f.write(str(auc)+"\n")
    #     #early stop
    #     if optimizer.param_groups[0]['lr'] < args.lr_early_stop:
    #         logloss,auc = ctr_evaluate(model, train_user_lists, train_item_lists, test_user_lists, neg_lists, args.n_shot, args, args.output_dim)
    #         print("finish training")
    #         with open(args.test_val_file,"a+") as f:
    #             f.write("test auc:"+str(auc)+" best val auc:"+str(best_auc))
    #         torch.save(model.state_dict(), "./checkpoint/best_{}.pt".format(args.model))
    #         break
    #     #adjust lr
    #     # scheduler.step(auc)
    #     #best performance in val
    #     if auc>=best_auc:
    #         best_auc = auc
    #     if val_less_time>args.early_stop_epoches:
    #         logloss,auc = ctr_evaluate(model, train_user_lists, train_item_lists, test_user_lists, neg_lists, args.n_shot, args, args.output_dim)
    #         print("test auc:",auc)
    #         torch.save(model.state_dict(), "./checkpoint/final.pt")




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
        
        # items = []

        # labels = []
        # adj_lists_temp = train_user_lists.copy()
        # adj_lists_temp_n_k = collections.defaultdict(list)
        # for user, items in test_user_lists.items():
        #     for item in items:
        #         train_item_lists[item].append(user)
        # item_lists_sample = train_item_lists.copy()
        # for user in list(test_user_lists.keys()):
        #     pos_items = list(np.random.choice(test_user_lists[user], k_shot,replace=False))
        #     # remove items in item_lists
        #     remove_items = set(test_user_lists[user])-set(pos_items)
        #     for remove_item in remove_items:
        #         adj_lists_temp_n_k[user].append(remove_item)
        #         temp = copy.copy(item_lists_sample[remove_item])
        #         temp.remove(user)
        #         item_lists_sample[remove_item] = temp
        #     test_user_lists[user] = pos_items
        #     neg_items = list(np.random.choice(neg_lists[user], k_shot,replace=False))
        #     label_user = [1]*k_shot+[0]*k_shot
        #     pos_items.extend(neg_items)
        #     # TODO: check 'items' and 'test_item_lists'
        #     items.append(pos_items)
        #     labels.append(label_user)

        # adj_lists_temp.update(test_user_lists)

        # so far, adj_lists_temp and item_lists_temp serves as test data

        # logloss_list = []
        # roc_auc_list = []

        # for user in list(test_user_lists.keys()):
        #     model_tmp = copy.deepcopy(model)
        #     model_tmp.train()
        #     optimizer_tmp = torch.optim.Adam(model_tmp.parameters(),lr=args.inner_lr)
        #     for inner_batch in range(args.inner_batch):
        #         # adj_lists_sample.update(user_lists_test)
        #         # pos_items,neg_items,adj_lists_sample = get_train_batch(users,train_user_lists,train_item_lists,k_shot,num_users,num_items)
        #         # items,labels,adj_lists_sample = get_train_batch_ctr(users,train_user_lists,train_item_lists,k_shot,neg_lists)
        #         labels = torch.tensor(labels).cuda()
        #         optimizer_tmp.zero_grad()
        #         # loss = model_tmp.loss(users,pos_items,neg_items,adj_lists_sample,few_shot=k_shot)
        #         loss = model_tmp.loss_ctr([user],items,labels,adj_lists_temp, few_shot=k_shot)
        #         loss.backward()
        #         optimizer_tmp.step()
        #     model_tmp.eval()
        #     user_embedding = model_tmp.forward([user], adj_lists_temp_n_k)
        #     user_embedding = torch.unsqueeze(user_embedding, 1).repeat([1, 2*few_shot, 1])
        #     items = np.reshape(items, -1)
        #     item_embedding = model_tmp.forward(items, adj_lists_temp_n_k)
        #     item_embedding = item_embedding.view([-1, 2*few_shot, self.output_dim])
        #     pred = torch.sum(item_embedding * user_embedding, -1).view(-1)
        #     labels = labels.view(-1)

        #     test_logloss = log_loss(labels, pred)
        #     logloss_list.append(test_logloss)
        #     test_roc_auc = roc_auc_score(labels, pred)
        #     roc_auc_list.append(test_roc_auc)
        #     print("Logloss: " , logloss_list, "ROC AUC Score: ", roc_auc_list)

        # print("Logloss: " , logloss_list.mean(), "ROC AUC Score: ", roc_auc_list.mean())