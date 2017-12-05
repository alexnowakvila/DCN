import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.spatial import ConvexHull

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F


baselines = ["DP_Kmeans_baseline4", "DP_Kmeans_baseline8",
             "DP_Kmeans_baseline16"]
DCN = ["DP_Kmeans_4", "DP_Kmeans_8", "DP_Kmeans_16"]

path = '/home/anowak/DynamicProgramming/DP/plots/'
n = [40, 80, 160]
ks = [4, 8, 16]

names = ['baseline', 'DCN']
for k in range(3):
    plt.figure(k)
    plt.clf()
    npzfile = np.load(path + baselines[k] + '/results.npz')
    reward_bl = npzfile['cost_kmeans']
    npzfile = np.load(path + DCN[k] + '/results.npz')
    reward_dcn = npzfile['cost_kmeans']
    l0, = plt.plot(reward_bl, c='b')
    l1, = plt.plot(reward_dcn, c='r')
    plt.xlabel('iterations')
    plt.ylabel('k_means cost')
    plt.title('n = {}, k = {}'.format(n[k], ks[k]))
    plt.legend([l0, l1], names, loc=1)
    plt.savefig(path + 'kmeans{}'.format(k) + '.png')
