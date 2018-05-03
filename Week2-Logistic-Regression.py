import matplotlib.pyplot as plt
import numpy as np
import torch

# Loading Data
with open('data.txt','r') as f:
    data_list=f.readlines()
    data_list=[i.split('\n') for i in data_list]
    data_list=[i.split(',') for i in data_list]
    data_list=[(float(i[0]),float(i[1]),float(i[2])) for i in data_list]




