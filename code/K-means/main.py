#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
import os
# import dependencies
from data_generator import Generator
from Logger import Logger
from DCN import DivideAndConquerNetwork
import utils
import time
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

#Pytorch requirements
import unicodedata
import string
import re
import random
import argparse
import pdb

import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

path = '/home/anowak/tmp/kmeans/exp4/'
path_dataset = '/data/anowak/kmeans/'
path_model = '/data/anowak/kmeans/models/exp4/'
# path_load_model = '/data/anowak/kmeans/models/exp2/'
path_load_model = None
parser = argparse.ArgumentParser()

parser.add_argument('--dynamic', action='store_true',
                    help='Use DCN. If not set, run baseline. (depth=0)')
parser.add_argument('--num_clusters', nargs='?', const=1, type=int, default=8)
parser.add_argument('--num_examples_train', nargs='?', const=1, type=int,
                    default=131072)
parser.add_argument('--num_examples_test', nargs='?', const=1, type=int,
                    default=4096)
parser.add_argument('--num_epochs', nargs='?', const=1, type=int, default=35)
parser.add_argument('--batch_size', nargs='?', const=1, type=int, default=256)
parser.add_argument('--mode', nargs='?', const=1, type=str, default='train')
parser.add_argument('--path_dataset', nargs='?', const=1,
                    type=str, default=path_dataset)
parser.add_argument('--path', nargs='?', const=1, type=str, default=path)


###############################################################################
#                              split arguments                                #
###############################################################################

parser.add_argument('--path_model', nargs='?', const=1, type=str,
                    default=path_model)
parser.add_argument('--path_load_model', nargs='?', const=1, type=str,
                    default=path_load_model)
parser.add_argument('--split_layers', nargs='?', const=1, type=int, default=5)
parser.add_argument('--num_units_split', nargs='?', const=1, type=int,
                    default=15)
parser.add_argument('--grad_clip_split', nargs='?', const=1,
                    type=float, default=40.0)
parser.add_argument('--regularize_split', action='store_true',
                    help='regularize the split training with variance prior.')
parser.add_argument('--beta', nargs='?', const=1, type=float, default=1.0)

args = parser.parse_args()

num_epochs = args.num_epochs
batch_size = args.batch_size
input_size = 2


if torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor
    dtype_l = torch.cuda.LongTensor
    torch.cuda.manual_seed(0)
else:
    dtype = torch.FloatTensor
    dtype_l = torch.LongTensor
    torch.manual_seed(0)


def train(DCN, logger, gen):
    Loss, Loss_reg = [], []
    Accuracies_tr = [[] for ii in gen.clusters['train']]
    Accuracies_te = [[] for ii in gen.clusters['test']]
    iterations_tr = int(gen.num_examples_train / batch_size)
    for epoch in range(num_epochs):
        for it in range(iterations_tr):
            losses = 0.0
            variances = 0.0
            start = time.time()
            for i, cl in enumerate(gen.clusters['train']):
                # depth tells how many times the dynamic model will be unrolled
                depth = np.log2(cl).astype(int)
                _, length = gen.compute_length(cl)
                input, _ = gen.get_batch(batch=it, clusters=cl,
                                         mode='train')
                # forward DCN
                out = DCN(input, length, depth, it=it, epoch=epoch,
                          mode='train', dynamic=True)
                Phis, Inputs_N, e, loss, pg_loss, var = out
                # backward DCN
                DCN.step_split(pg_loss, var, regularize=args.regularize_split)
                losses += loss
                variances += var
                Accuracies_tr[i].append(loss.data.cpu().numpy())
                # ll = utils.Lloyds(input, n_clusters=cl)
                # pdb.set_trace()
            losses /= len(gen.clusters['train'])
            variances /= len(gen.clusters['train'])
            Loss.append(losses.data.cpu().numpy())
            Loss_reg.append(variances.data.cpu().numpy())
            elapsed = time.time() - start
            if it % 64 == 0:
                print('TRAINING --> | Epoch {} | Batch {} / {} | Loss {} |'
                      ' Elapsed {} | dynamic {} |'
                      .format(epoch, it, iterations_tr,
                              losses.data.cpu().numpy(),
                              elapsed, args.dynamic))
                logger.plot_Phis_sparsity(Phis, fig=0)
                logger.plot_losses(Loss, Loss_reg, fig=2)
                logger.plot_classes(input, cl, e, fig=cl)
        if epoch % 5 == 4:
            print('Saving model parameters')
            DCN.save_split(args.path_model)
        accuracies_test = test(DCN, gen)
        for i, cl in enumerate(gen.clusters['test']):
            Accuracies_te[i].append(accuracies_test[i])
        show_accs = ' | '.join(['Acc Len {} -> {}'
                                .format(cl, Accuracies_te[i][-1])
                                for i, cl
                                in enumerate(gen.clusters['test'])])
        print('TESTING --> epoch {} '.format(epoch) + show_accs)
        logger.save_results(Accuracies_te)


def test(DCN, gen):
    accuracies_test = [[] for ii in gen.clusters['test']]
    iterations_te = int(gen.num_examples_test / batch_size)
    for it in range(iterations_te):
        for i, cl in enumerate(gen.clusters['test']):
            # depth tells how many times the dynamic model will be unrolled
            depth = np.log2(cl).astype(int)
            _, length = gen.compute_length(cl)
            input, _ = gen.get_batch(batch=it, clusters=cl,
                                     mode='test')
            # forward DCN
            out = DCN(input, length, depth, it=it,
                      mode='test', dynamic=True)
            Phis, Inputs_N, e, loss, pg_loss, var = out
            cost = utils.cost(input, e.data.cpu().numpy(),
                              n_clusters=args.num_clusters)
            accuracies_test[i].append(cost)
    accuracies_test = [sum(accs) / iterations_te
                       for accs in accuracies_test]
    return accuracies_test

if __name__ == '__main__':
    logger = Logger(args.path)
    logger.write_settings(args)
    DCN = DivideAndConquerNetwork(input_size, args.batch_size,
                                  args.num_units_split, args.split_layers,
                                  args.grad_clip_split, beta=args.beta)
    if args.path_load_model is not None:
        DCN.load_split(args.path_load_model)
    if torch.cuda.is_available():
        DCN.cuda()
    gen = Generator(args.num_examples_train, args.num_examples_test,
                    args.num_clusters, args.path_dataset, args.batch_size)
    gen.load_dataset()
    if args.mode == 'train':
        train(DCN, logger, gen)
    elif args.mode == 'test':
        test(DCN, logger, gen)
