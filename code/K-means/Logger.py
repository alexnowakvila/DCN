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


class Logger(object):
    def __init__(self, path):
        directory = os.path.join(path, 'plots/')
        self.path = directory
        # Create directory if necessary
        try:
            os.stat(directory)
        except:
            os.mkdir(directory)

    def write_settings(self, args):
        # write info
        path = self.path + 'settings.txt'
        with open(path, 'w') as file:
            for arg in vars(args):
                file.write(str(arg) + ' : ' + str(getattr(args, arg)) + '\n')

    def plot_losses(self, losses, reward, scales=[], fig=0):
        # discriminative losses
        plt.figure(fig)
        plt.clf()
        plt.semilogy(range(0, len(losses)), losses, 'b')
        plt.xlabel('iterations')
        plt.ylabel('Loss')
        plt.title('loss')
        path = os.path.join(self.path, 'losses.png')
        plt.savefig(path)
        # reward
        plt.figure(fig)
        plt.clf()
        plt.plot(range(0, len(reward)), reward, 'b')
        plt.xlabel('iterations')
        plt.title('kmeans cost')
        path = os.path.join(self.path, 'cost.png')
        plt.savefig(path)

    def plot_classes(self, points, clusters, e, fig=0):
        e = e[0].data.cpu().numpy()
        points = points[0]
        plt.figure(fig)
        plt.clf()
        colors = cm.rainbow(np.linspace(0, 1, clusters))
        for cl in xrange(clusters):
            ind = np.where(e == cl)[0]
            pts = points[ind]
            plt.scatter(pts[:, 0], pts[:, 1], c=colors[cl])
        plt.title('clustering')
        path = os.path.join(self.path, 'clustering_ex.png'.format(clusters))
        plt.savefig(path)

    def plot_Phis_sparsity(self, Phis, fig=0):
        Phis = [phis[0].data.cpu().numpy() for phis in Phis]
        plt.figure(fig)
        plt.clf()
        for i, phi in enumerate(Phis):
            plt.subplot(1, len(Phis), i + 1)
            # plot first element of the batch
            plt.spy(phi, precision=0.001, marker='o', markersize=2)
            plt.xticks([])
            plt.yticks([])
            plt.title('k={}'.format(i))
        path = os.path.join(self.path, 'Phis.png')
        plt.savefig(path)

    def save_results(self, cost_kmeans):
        np.savez(self.path + 'results.npz', cost_kmeans=cost_kmeans)

    def save_test_results(self, accuracies_test, Discard_rates, scales):
        path = self.path + 'test_results.txt'
        with open(path, 'w') as file:
            file.write('--------------TEST RESULTS-------------- \n')
            for i, accs in enumerate(accuracies_test):
                result_acc = ('Accuracy for {} scales: {} \n'
                              .format(scales[i], accs))
                result_rate = ('Discard rates for {} scales:'
                               .format(scales[i]))
                rates_str = ', '.join([str(rate) for rate in Discard_rates[i]])
                file.write(result_acc + result_rate + rates_str + '\n')
