import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
import torch.nn.functional as F

import numpy as np
import time
import random
from collections import defaultdict

class gcn(nn.Module):
    def __init__(self,features,input_dim=64,hidden_dim=64,output_dim=64, num_layers=2,dropout_rate=0.1):
        super(gcn, self).__init__()
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
        pass
        #todo

    def forward(self,nodes,adj_lists,flag=False):
        #todo
        pass









