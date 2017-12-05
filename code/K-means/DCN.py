#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
import os
# import dependencies
from Split import Split
# import utils
import time
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

#Pytorch requirements
import unicodedata
import string
import re
import random
import pdb
import argparse

import torch
import torch.nn as nn
from torch.nn import init
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


class DivideAndConquerNetwork(nn.Module):
    def __init__(
                 self, input_size, batch_size,
                 num_units_split, split_layers, grad_clip_split, beta=1.0
                 ):
        super(DivideAndConquerNetwork, self).__init__()
        # General
        self.input_size = input_size
        self.batch_size = batch_size
        # Split
        self.num_units_split = num_units_split
        self.split_layers = split_layers
        self.beta = beta
        self.split = Split(input_size, num_units_split,
                           batch_size, split_layers)
        # Training
        self.grad_clip_split = grad_clip_split
        self.optim_split = optim.RMSprop(self.split.parameters())

    ###########################################################################
    #                           Load Parameters                               #
    ###########################################################################

    def load_split(self, path):
        path = os.path.join(path, 'parameters/params_split.pt')
        if os.path.exists(path):
            self.split = torch.load(path)
        else:
            raise ValueError('path for split {} does not exist'.format(path))

    ###########################################################################
    #                           Save Parameters                               #
    ###########################################################################

    def save_split(self, path):
        try: 
            os.stat(path) 
        except: 
            os.mkdir(path)
        directory = os.path.join(path, 'parameters/')
        try: 
            os.stat(directory) 
        except: 
            os.mkdir(directory)
        path = os.path.join(directory, 'params_split.pt')
        torch.save(self.split, path)

    ###########################################################################
    #                        Optimization Steps                               #
    ###########################################################################

    def step_split(self, cost, variances, regularize=False):
        self.split.zero_grad()
        loss = cost
        if regularize:
            loss -= self.beta * variances
        loss.backward()
        nn.utils.clip_grad_norm(self.split.parameters(), self.grad_clip_split)
        self.optim_split.step()

    ###########################################################################
    #                           Compute Rewards                               #
    ###########################################################################

    def compute_diameter(self, input, e, cl, it=0):
        m1 = (e == 2 * cl).type(dtype)
        n1 = m1.sum(1)
        n1 += (n1 == 0).type(dtype)
        m2 = (e == 2 * cl + 1).type(dtype)
        n2 = m2.sum(1)
        n2 += (n2 == 0).type(dtype)
        m1 = m1.unsqueeze(2).expand_as(input)
        m2 = m2.unsqueeze(2).expand_as(input)
        n1 = n1.unsqueeze(1).expand(self.batch_size, 2)
        n2 = n2.unsqueeze(1).expand(self.batch_size, 2)
        mean1 = (input * m1).sum(1) / n1
        centered1 = (input - mean1.unsqueeze(1).expand_as(input)) * m1
        vars1 = (centered1 * centered1).sum(1)
        var1 = vars1.squeeze().squeeze().sum(1)
        mean2 = (input * m2).sum(1) / n2
        centered2 = (input - mean2.unsqueeze(1).expand_as(input)) * m2
        vars2 = (centered2 * centered2).sum(1)
        var2 = vars2.squeeze().sum(1)
        return var1 + var2, m1[:, :, 0].squeeze(), m2[:, :, 0].squeeze()

    def compute_loss(self, input, e, b, clusters, it=0):
        Loss = Variable(torch.zeros((self.batch_size))).type(dtype)
        Ls = Variable(torch.zeros((self.batch_size))).type(dtype)
        for cl in range(clusters // 2):
            L, m1, m2 = self.compute_diameter(input, e, cl, it=it)
            mask = ((e / 2).type(dtype_l) == cl).type(dtype)
            # print('mask', mask[0])
            n = mask.sum(1).squeeze()
            n += (n == 0).type(dtype)
            # print('mask', mask[0])
            log_probs = torch.log((1 - b) * m1 + b * m2 + (1 - mask) + 1e-8)
            Loss += L * log_probs.sum(1) / n
            Ls += L
        Ls = Ls.mean(0)
        Loss = Loss.mean(0)
        return Loss, Ls

    ###########################################################################
    #                             Split Phase                                 #
    ###########################################################################

    def log_probabilities(self, Bs, Samples, mask, depth):
        LogProbs = []
        lengths = mask.sum(1)
        for scale in range(depth):
            probs = Bs[scale]
            sample = Samples[scale]
            probs_act = probs * sample + (1 - probs) * (1 - sample)
            logprobs = torch.log(probs_act + (1 - mask) + 1e-6)
            logprobs = logprobs.sum(1) / lengths
            LogProbs.append(logprobs)
        return LogProbs

    def compute_variance(self, probs, mask):
        N = mask.sum(1)
        mean = torch.sum(probs, 1) / N
        mean_squared = mean * mean
        mean_std = torch.mean(torch.sum(probs*probs, 1) / N -
                              mean_squared, 0)
        return mean_std.squeeze()

    def fwd_split(self, input, batch, depth,
                  mergesort_split=False, mode='train', epoch=0):
        length = self.split.n
        var = 0.0
        # Iterate over scales
        e = Variable(torch.zeros((self.batch_size, length)).type(dtype),
                     requires_grad=False)
        mask = (input[:, :, 0] >= 0).type(dtype).squeeze()
        Phis, Bs, Inputs_N, Samples = ([] for ii in range(4))
        for scale in range(depth):
            logits, probs, input_n, Phi = self.split(e, input,
                                                     mask, scale=scale)
            # Sample from probabilities and update embeddings
            rand = (Variable(torch.zeros(self.batch_size, length))
                    .type(dtype))
            init.uniform(rand)
            sample = (probs > rand).type(dtype)
            e = 2 * e + sample
            # Appends
            Samples.append(sample)
            Phis.append(Phi)
            Bs.append(probs)
            Inputs_N.append(input_n)
            # variance of bernouilli probabilities
            var += self.compute_variance(probs, mask)
        # computes log probabilities of binary actions for the policy gradient
        Log_Probs = self.log_probabilities(Bs, Samples, mask, depth)
        # pad embeddings with infinity to not affect embeddings argsort
        infty = 1e6
        e = e * mask + (1 - mask) * infty
        return var, Phis, Bs, Inputs_N, e, Log_Probs

    ###########################################################################
    #                            Forward pass                                 #
    ###########################################################################

    def forward(self, input, length, depth, it=0, epoch=0,
                mergesort_split=False, mode='train', dynamic=False):
        self.split.n = length
        input = (Variable(torch.from_numpy(input), requires_grad=False)
                 .type(dtype))
        # input = input[0].unsqueeze(0).expand_as(input)
        # forward split
        out_split = self.fwd_split(input, it, depth,
                                   mergesort_split=mergesort_split,
                                   mode=mode, epoch=epoch)
        var, Phis, Bs, Inputs_N, e, lp = out_split
        # compute reward and policy gradient loss
        pg_loss = 0.0
        for j, scale in enumerate(range(depth)):
            div = 2 ** (depth - scale - 1)
            cl_scale = 2 ** (scale + 1)
            samples = (e / float(div)).type(dtype_l)
            lrwd, acc = self.compute_loss(Inputs_N[0], samples, Bs[j],
                                          cl_scale, it=it)
            pg_loss = lrwd
        loss = acc
        return Phis, Inputs_N, e, loss, pg_loss, var
