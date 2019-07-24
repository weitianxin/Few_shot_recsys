import numpy as np
import random as rd
import scipy.sparse as sp
from time import time
import json

def load_json(json_dir='./data/yelp_dataset/yelp_dataset'):
    json_dir = './data/yelp_dataset/yelp_dataset'
    with open(json_dir) as f:
        load_dict = json.load(f)
        print(load_dict)
    #todo
