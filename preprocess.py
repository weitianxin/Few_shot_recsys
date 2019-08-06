import matplotlib.pyplot as plt
import numpy
import json
import collections
import seaborn as sns
import numpy as np
#read dataset
with open("dataset/reviews_Electronics_5.json") as f:
    data = [json.loads(line) for line in f]
    interactions = [(sample['reviewerID'],sample['asin']) for sample in data]
user_lists = collections.defaultdict(list)
item_lists = collections.defaultdict(list)
for interaction in interactions:
    user,item = interaction[0],interaction[1]
    user_lists[user].append(item)
    item_lists[item].append(user)
print(len(user_lists), len(item_lists))
#filter
K = 10
i=1
while i!=0:
    i=0
    delete_list = []
    for user,items in user_lists.items():
        if len(items)<K:
            i += 1
            for item in items:
                item_lists[item].remove(user)
            delete_list.append(user)
    for delete in delete_list:
        user_lists.pop(delete)
    delete_list = []
    for item,users in item_lists.items():
        if len(users)<K:
            i += 1
            for user in users:
                user_lists[user].remove(item)
            delete_list.append(item)
    for delete in delete_list:
        item_lists.pop(delete)
print(len(user_lists), len(item_lists))
user_num,item_num = len(user_lists), len(item_lists)
#reid
user_old2new,item_old2new = {},{}
id = 0
for user in user_lists.keys():
    user_old2new[user] = id
    id+=1
id = len(user_old2new)
for item in item_lists.keys():
    item_old2new[item] = id
    id+=1
#reform dataset
user_lists_id = collections.defaultdict(list)
item_lists_id = collections.defaultdict(list)
for user,items in user_lists.items():
    user_id = user_old2new[user]
    for item in items:
        user_lists_id[user_id].append(item_old2new[item])

for item,users in item_lists.items():
    item_id = item_old2new[item]
    for user in users:
        item_lists_id[item_id].append(user_old2new[user])
#neg sample
neg_lists = collections.defaultdict(list)
for user,items in user_lists_id.items():
    neg_items = []
    for i in range(len(items)):
        flag = True
        while flag:
            neg = np.random.randint( user_num,user_num+item_num)
            if neg not in user_lists_id[user] and neg not in neg_items:
                neg_items.append(neg)
                flag=False
    neg_lists[user].extend(neg_items)

with open("dataset/amazon_elec/user_lists_10","w") as f:
    f.write(str(dict(user_lists_id)))

with open("dataset/amazon_elec/item_lists_10","w") as f:
    f.write(str(dict(item_lists_id)))

with open("dataset/amazon_elec/neg_lists_10","w") as f:
    f.write(str(dict(neg_lists)))
