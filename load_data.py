import collections
import time
import scipy.sparse as sp
import torch
import numpy as np
import random


def amazon_ctr(ratio=0.8,dir="dataset/amazon_elec/"):
    with open(dir+"user_lists_10") as f:
        user_lists = eval(f.read())
    # with open(dir+"item_lists_10") as f:
    #     item_lists = eval(f.read())
    with open(dir+"neg_lists_10") as f:
        neg_lists = eval(f.read())
    users = list(user_lists.keys())
    n_users = len(users)
    #split dataset
    random.seed(2019)
    random.shuffle(users)
    bound = int(ratio * len(users))
    test_val_ratio = (1-ratio)/2
    test_val_size = int(test_val_ratio * len(users))
    train_index = users[:bound]
    val_index = users[bound:bound+1]
    test_index = users[bound+test_val_size:]
    train_user_lists = {i: user_lists[i] for i in train_index}
    val_user_lists = {i: user_lists[i] for i in val_index}
    test_user_lists = {i: user_lists[i] for i in test_index}
   
    # build item lists
    train_item_lists = collections.defaultdict(list)
    for user, items in train_user_lists.items():
        for item in items:
            train_item_lists[item].append(user)
    n_items = len(train_item_lists)
    return neg_lists,train_user_lists, val_user_lists, test_user_lists, train_item_lists, n_users, n_items


def gowalla(ratio=0.8):
    data_dir = './dataset/'
    train_dir = data_dir + "train.txt"
    test_dir = data_dir + "test.txt"
    user_lists = collections.defaultdict(list)
    train_item_lists = collections.defaultdict(list)
    n_items = 40981
    n_users = 29858
    with open(train_dir) as f:
        for l in f.readlines():
            if len(l) > 0:
                l = l.strip('\n').split(' ')
                items = [29858+int(i) for i in l[1:]]
                uid = int(l[0])
                user_lists[uid].extend(items)

    with open(test_dir) as f:
        for l in f.readlines():
            if len(l) > 0:
                l = l.strip('\n').split(' ')
                items = [29858+int(i) for i in l[1:]]
                uid = int(l[0])
                user_lists[uid].extend(items)

    users = list(user_lists.keys())
    random.seed(2019)
    random.shuffle(users)
    bound = int(ratio*n_users)
    test_val_ratio = (1-ratio)/2
    test_val_size = int(test_val_ratio * len(users))
    train_index = users[:bound]
    val_index = users[bound:bound+1]
    test_index = users[bound+test_val_size:]

    train_user_lists = {i: user_lists[i] for i in train_index}
    val_user_lists = {i: user_lists[i] for i in val_index}
    test_user_lists = {i: user_lists[i] for i in test_index}
    #build item lists
    for user,items in train_user_lists.items():
        for item in items:
            train_item_lists[item].append(user)
    #neg_lists
    neg_lists = collections.defaultdict(list)
    for user,items in user_lists.items():
        neg_items = []
        for i in range(len(items)):
            flag = True
            while flag:
                neg = np.random.randint( n_users,n_users+n_items)
                if neg not in user_lists[user] and neg not in neg_items:
                    neg_items.append(neg)
                    flag=False
        neg_lists[user].extend(neg_items)
    return neg_lists,train_user_lists, val_user_lists, test_user_lists, train_item_lists, n_users, n_items

