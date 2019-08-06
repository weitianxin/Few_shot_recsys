import numpy as np
import copy
def get_train_gnn_batch_ctr(users,user_lists,item_lists, neg_lists):
    pass
    
def get_train_batch_ctr(users,user_lists,item_lists,k_shot, neg_lists):
    adj_lists_sample = user_lists.copy()
    item_lists_sample = item_lists.copy()
    items = []
    labels = []
    for user in users:
        pos_items = list(np.random.choice(user_lists[user], k_shot,replace=False))
        # remove items in item_lists
        remove_items = set(user_lists[user])-set(pos_items)
        for remove_item in remove_items:
            temp = copy.copy(item_lists_sample[remove_item])
            temp.remove(user)
            item_lists_sample[remove_item] = temp
        adj_lists_sample[user] = pos_items
        neg_items = list(np.random.choice(neg_lists[user], k_shot,replace=False))
        label_user = [1]*k_shot+[0]*k_shot
        pos_items.extend(neg_items)
        #是否shuffle有影响嘛？？？
        items.append(pos_items)
        labels.append(label_user)

    adj_lists_sample.update(item_lists_sample)
    return items,labels,adj_lists_sample


def get_test_batch_ctr(users,adj_lists_temp,k_shot,neg_lists):
    items = []
    labels = []
    for user in users:
        pos_items = list(adj_lists_temp[user])
        neg_items = list(neg_lists[user][:k_shot])
        label_user = [1]*k_shot+[0]*k_shot
        pos_items.extend(neg_items)
        items.append(pos_items)
        labels.append(label_user)

    return items,labels


def sample_test_item(user, adj_lists,neg_lists,k_shot):
    # pos items
    items = adj_lists[user]
    n_k = len(items)
    # neg items
    neg_items = neg_lists[user][k_shot:]
    items.extend(neg_items)
    labels = [1]*n_k+[0]*n_k
    return items, n_k, labels


def get_train_batch(users,user_lists,item_lists,k_shot, n_users, n_items):
    adj_lists_sample = user_lists.copy()
    item_lists_sample = item_lists.copy()
    pos_list,neg_list = [],[]
    for user in users:
        pos_items = list(np.random.choice(user_lists[user], k_shot,replace=False))
        # remove items in item_lists
        remove_items = set(user_lists[user])-set(pos_items)
        for remove_item in remove_items:
            temp = copy.copy(item_lists_sample[remove_item])
            temp.remove(user)
            item_lists_sample[remove_item] = temp
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
