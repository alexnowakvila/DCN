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

if torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor
    dtype_l = torch.cuda.LongTensor
    torch.cuda.manual_seed(0)
else:
    dtype = torch.FloatTensor
    dtype_l = torch.LongTensor
    torch.manual_seed(0)


def build_phi(e):
    e1 = np.reshape(e, [-1, 1])
    e2 = np.reshape(e, [1, -1])
    return (e1 == e2).astype(float)


def split_recursively(b, ind, scale, scales):
    if scale == scales:
        return b
    else:
        l = len(ind)
        ind1 = ind[:(l // 2)]
        ind2 = ind[(l // 2):]
        b[ind1, scale] = 0
        b = split_recursively(b, ind1, scale + 1, scales)
        b[ind2, scale] = 1
        b = split_recursively(b, ind2, scale + 1, scales)
        return b


def mergesort(lengths, max_length, scales):
    batch_size = lengths.shape[0]
    Phi = [np.zeros([batch_size, max_length, max_length])
           for scale in xrange(scales)]
    B = [np.zeros([batch_size, max_length]) for scale in xrange(scales)]
    E = np.zeros([batch_size, max_length])
    for batch in xrange(batch_size):
        length = lengths[batch]
        b = np.zeros([length, scales])
        ind = range(length)
        b = split_recursively(b, ind, 0, scales)
        e = np.zeros([length])
        for scale in xrange(scales):
            Phi[scale][batch, :length, :length] = build_phi(e)
            e = 2 * e + b[:, scale]
            B[scale][batch, :length] = b[:, scale]
        E[batch, :length] = e
    return E, B, Phi


def mergesort_split(mask, scales):
    lengths = np.sum(mask, axis=1).astype(int)
    max_length = mask.shape[1]
    e, b, phi = mergesort(lengths, max_length, scales)
    E = Variable(torch.from_numpy(e)).type(dtype)
    B = [Variable(torch.from_numpy(bb)).type(dtype) for bb in b]
    Phi = [Variable(torch.from_numpy(ph)).type(dtype) for ph in phi]
    return E, B, Phi


def compute_accuracy(output, target, it=0):
    tar = target.data.cpu().numpy()
    out = output.data.cpu().numpy()
    batch_size = tar.shape[0]
    accuracy = 0.0
    for k in xrange(batch_size):
        accuracy += float(np.sum(tar[k] != out[k]) == 0)
    return accuracy / float(batch_size)


def pad_eye(Phis):
    mask = Phis[0]
    length = mask.size()[1]
    pad = Variable(torch.eye(length, length)).type(dtype)
    pad = pad.expand_as(mask) * (1 - mask)
    for i, phis in enumerate(Phis):
        Phis[i] = Phis[i] + pad
    return Phis
