import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
import torch.nn.functional as F

import numpy as np
import time
import random
from collections import defaultdict

class graphsage(nn.Module):
    def __init__(self,features,input_dim=64,hidden_dim=64,output_dim=64, num_layers=2,dropout_rate=0.1):
        super(graphsage, self).__init__()
        self.features = features
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        #hypermeter
        self.dropout_rate = dropout_rate
        #build weight
        self.linear_list = nn.ModuleList()
        self.dropout_list = nn.ModuleList()
        self.linear_list.append(nn.Linear(input_dim,hidden_dim))
        self.dropout_list.append(nn.Dropout(p=dropout_rate))
        for i in range(num_layers-2):
            self.linear_list.append(nn.Linear(hidden_dim, hidden_dim))
            self.dropout_list.append(nn.Dropout(p=dropout_rate))
        self.linear_list.append(nn.Linear(hidden_dim, output_dim))
        self.dropout_list.append(nn.Dropout(p=dropout_rate))
        self.transform = nn.LogSigmoid()
        #build dropout

    def loss(self,users,pos_items,neg_items,adj_lists,few_shot=2):
        #users:batch
        user_embedding = self.forward(users,adj_lists)

        user_embedding = torch.unsqueeze(user_embedding,1).repeat([1,few_shot,1])
        #items: batch_size x k
        pos_items = np.reshape(pos_items,-1)
        neg_items = np.reshape(neg_items,-1)

        pos_item_embedding = self.forward(pos_items,adj_lists,flag=True)

        neg_item_embedding = self.forward(neg_items,adj_lists)
        #batch x k x embed_dim
        pos_item_embedding = pos_item_embedding.view([-1,few_shot,self.output_dim])
        neg_item_embedding = neg_item_embedding.view([-1,few_shot,self.output_dim])
        #result
        pos = torch.sum(pos_item_embedding*user_embedding,-1)
        neg = torch.sum(neg_item_embedding*user_embedding,-1)
        #输入规模为1 x
        loss = -torch.sum(self.transform(pos-neg))/few_shot
        return loss

    def forward(self,nodes,adj_lists,flag=False):
        nodes_all = self.get_neighbor(nodes,adj_lists)
        vectors = [self.features(torch.tensor(node_layer).cuda()) for node_layer in nodes_all]
        for i in range(self.num_layers):
            vectors_next_iter = []
            for hop in range(self.num_layers - i):
                #!!!!
                vector = self.aggregate([adj_lists[node] for node in nodes_all[hop]],nodes_all[hop+1],vectors[hop+1],
                                        i)
                vectors_next_iter.append(vector)
            vectors = vectors_next_iter
        return vectors[0]


    def aggregate(self,samp_neighs,neig_nodes,neig_vectors,i):
        unique_nodes_list = neig_nodes
        linear = self.linear_list[i]
        dropout = self.dropout_list[i]
        # 在原图中的index对应当前采样index的字典
        # time1 = time.time()
        unique_nodes = {n: i for i, n in enumerate(unique_nodes_list)}
        mask = torch.zeros(len(samp_neighs), len(unique_nodes))
        column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]
        row_indices = [i for i in range(len(samp_neighs)) for j in range(len(samp_neighs[i]))]
        mask[row_indices, column_indices] = 1
        # time2 = time.time()
        num_neigh = mask.sum(1, keepdim=True)
        mask = mask.div(num_neigh)
        embed_matrix = neig_vectors.cuda()
        # time3 = time.time()
        # mask = mask.to_sparse()
        # to_feats = torch.sparse.mm(mask,embed_matrix)
        mask = mask.cuda()
        to_feats = mask.mm(embed_matrix)
        to_feats = dropout(F.relu(linear(to_feats)))
        # time4 = time.time()
        # total = time4-time1
        # print((time2-time1)/total,(time3-time2)/total,(time4-time3)/total,total)
        return to_feats

    def get_neighbor(self,nodes,adj_lists,num_sample=None):
        #nodes_list: hop 0-n
        nodes_list = [nodes]
        for i in range(self.num_layers):
            temp_nodes = []
            if not num_sample:
                for node in nodes_list[i]:
                    temp = adj_lists[node]
                    # temp.append(node)
                    temp_nodes.extend(temp)
                neighbors = list(set(temp_nodes))
                nodes_list.append(neighbors)
            else:
                pass#todo
        return nodes_list


# class Encoder(nn.Module):
#     """
#     Encodes a node's using 'convolutional' GraphSage approach
#     """
#     def __init__(self, features, feature_dim,
#             embed_dim, adj_lists, aggregator,
#             num_sample=None, loop=True, cuda=True,
#             feature_transform=False):
#         super(Encoder, self).__init__()
#
#         self.features = features
#         self.feat_dim = feature_dim
#         self.adj_lists = adj_lists
#         self.aggregator = aggregator
#         self.num_sample = num_sample
#         #决定要不要self-loop
#         self.loop = loop
#         self.embed_dim = embed_dim
#         self.cuda = cuda
#         self.aggregator.cuda = cuda
#         self.weight = nn.Parameter(
#                 torch.FloatTensor(embed_dim, self.feat_dim if self.loop else 2 * self.feat_dim))
#         init.xavier_uniform(self.weight)
#
#     def forward(self, nodes,num_sample=None):
#         """
#         Generates embeddings for a batch of nodes.
#
#         nodes     -- list of nodes
#         """
#         #方便自己决定要计算embedding的
#         if num_sample:
#             self.num_sample = num_sample
#         neigh_feats = self.aggregator.forward(nodes, [self.adj_lists[int(node)] for node in nodes],
#                 self.num_sample)
#         if not self.loop:
#             if self.cuda:
#                 self_feats = self.features(torch.LongTensor(nodes).cuda())
#             else:
#                 self_feats = self.features(torch.LongTensor(nodes))
#             combined = torch.cat([self_feats, neigh_feats], dim=1)
#         else:
#             combined = neigh_feats
#         combined = F.relu(self.weight.mm(combined.t()))
#         return combined
#
#
# class Aggregator(nn.Module):
#     """
#     Aggregates a node's embeddings using mean of neighbors' embeddings
#     """
#
#     def __init__(self, features, cuda=True, loop=True):
#         """
#         Initializes the aggregator for a specific graph.
#
#         features -- function mapping LongTensor of node ids to FloatTensor of feature values.
#         cuda -- whether to use GPU
#         gcn --- whether to perform concatenation GraphSAGE-style, or add self-loops GCN-style
#         """
#
#         super(Aggregator, self).__init__()
#
#         self.features = features
#         self.cuda = cuda
#         self.loop = loop
#
#     def forward(self, nodes, to_neighs, num_sample):
#         """
#         nodes --- list of nodes in a batch
#         to_neighs --- list of sets, each set is the set of neighbors for node in batch
#         num_sample --- number of neighbors to sample. No sampling if None.
#         """
#         # Local pointers to functions (speed hack)
#         _set = set
#         if not num_sample is None:
#             _sample = random.sample
#             samp_neighs = [_set(_sample(to_neigh,
#                                         num_sample,
#                                         )) if len(to_neigh) >= num_sample else to_neigh for to_neigh in to_neighs]
#         else:
#             samp_neighs = to_neighs
#
#         if self.loop:
#             samp_neighs = [samp_neigh + set([nodes[i]]) for i, samp_neigh in enumerate(samp_neighs)]
#         #batch中所有Node需要计算的embedding
#         unique_nodes_list = list(set.union(*samp_neighs))
#         #在原图中的index对应当前采样index的字典
#         unique_nodes = {n: i for i, n in enumerate(unique_nodes_list)}
#
#         mask = Variable(torch.zeros(len(samp_neighs), len(unique_nodes)))
#         column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]
#         row_indices = [i for i in range(len(samp_neighs)) for j in range(len(samp_neighs[i]))]
#         mask[row_indices, column_indices] = 1
#         if self.cuda:
#             mask = mask.cuda()
#         num_neigh = mask.sum(1, keepdim=True)
#         mask = mask.div(num_neigh)
#         if self.cuda:
#             embed_matrix = self.features(torch.LongTensor(unique_nodes_list).cuda())
#         else:
#             embed_matrix = self.features(torch.LongTensor(unique_nodes_list))
#         to_feats = mask.mm(embed_matrix)
#         return to_feats
#
# class GraphSage(nn.Module):
#
#     def __init__(self,features,adj_lists,input_dim=64,hidden_dim=64,output_dim=64, layers=2):
#         super(GraphSage, self).__init__()
#         self.output_dim = output_dim
#         #first layer
#         agg_temp = Aggregator(features)
#         enc_temp = Encoder(features, input_dim, hidden_dim, adj_lists, agg_temp, loop=True)
#         for i in range(layers-2):
#             agg_temp = Aggregator(lambda nodes : enc_temp(nodes).t())
#             enc_temp = Encoder(lambda nodes : enc_temp(nodes).t(), enc_temp.embed_dim, hidden_dim, adj_lists, agg_temp,
#                                loop=True, cuda=True)
#
#         self.agg = Aggregator(lambda nodes : enc_temp(nodes).t())
#         self.enc = Encoder(lambda nodes : enc_temp(nodes).t(), enc_temp.embed_dim, output_dim, adj_lists, self.agg,
#                            loop=True, cuda=True)
#         self.xent = nn.CrossEntropyLoss()
#         self.transform = nn.LogSigmoid()
#
#
#     def forward(self, nodes,few_shot=None):
#         embeds = self.enc(nodes,num_sample=few_shot)
#         return embeds
#
#     def loss(self,users,pos_items,neg_items,few_shot=2):
#         users:batch
#         user_embedding = self.forward(users,few_shot=few_shot)
#         user_embedding = torch.unsqueeze(user_embedding,1).repeat([1,few_shot,1])
#         #items: batch_size x k
#         pos_items = pos_items.view(-1)
#         neg_items = neg_items.view(-1)
#         pos_item_embedding = self.forward(pos_items)
#         neg_item_embedding = self.forward(neg_items)
#         #batch x k x embed_dim
#         pos_item_embedding = pos_item_embedding.view([-1,few_shot,self.output_dim])
#         neg_item_embedding = neg_item_embedding.view([-1,few_shot,self.output_dim])
#         #result
#         pos = torch.sum(pos_item_embedding*user_embedding,-1)
#         neg = torch.sum(neg_item_embedding*user_embedding,-1)
#         loss = -torch.sum(self.transform(pos-neg),-1)
#         return loss






