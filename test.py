import networkx
import numpy as np
import time
import torch
a = torch.ones([7000,7000]).cuda()
b = torch.ones([7000,64]).cuda()
time1 = time.time()
for i in range(30):
    torch.mm(a,b)
time2 = time.time()
print(time2-time1)
