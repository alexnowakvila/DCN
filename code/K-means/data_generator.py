import numpy as np
import os
from scipy.spatial import ConvexHull
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import matplotlib.cm as cm


class Generator(object):
    def __init__(
                 self, num_examples_train, num_examples_test, num_clusters,
                 dataset_path, batch_size
                 ):
        self.num_examples_train = num_examples_train
        self.num_examples_test = num_examples_test
        self.batch_size = batch_size
        self.dataset_path = dataset_path
        self.input_size = 2
        self.task = 'kmeans'
        # clusters_train = [4, 8, 16]
        clusters_train = [num_clusters]
        clusters_test = [num_clusters]
        self.clusters = {'train': clusters_train, 'test': clusters_test}
        self.data = {'train': {}, 'test': {}}

    def load_dataset(self):
        for mode in ['train', 'test']:
            for cl in self.clusters[mode]:
                path = os.path.join(self.dataset_path, mode + str(cl))
                path = path + 'kmeans_gauss.npz'
                if os.path.exists(path):
                    print('Reading {} dataset for {} scales'
                          .format(mode, cl))
                    npz = np.load(path)
                    self.data[mode][cl] = {'x': npz['x'], 'y': npz['y']}
                else:
                    x, y = self.create(clusters=cl, mode=mode)
                    self.data[mode][cl] = {'x': x, 'y': y}
                    # save
                    np.savez(path, x=x, y=y)
                    print('Created {} dataset for {} scales'
                          .format(mode, cl))

    def get_batch(self, batch=0, clusters=3, mode="train"):
        bs = self.batch_size
        batch_x = self.data[mode][clusters]['x'][batch * bs: (batch + 1) * bs]
        batch_y = self.data[mode][clusters]['y'][batch * bs: (batch + 1) * bs]
        return batch_x, batch_y

    def compute_length(self, clusters):
        length = np.random.randint(10 * clusters, 10 * clusters + 1)
        max_length = 10 * clusters
        return length, max_length

    def kmeans_example(self, length, clusters):
        points = np.random.uniform(0, 1, [length, 2])
        kmeans = KMeans(n_clusters=clusters).fit(points)
        labels = kmeans.labels_.astype(int)
        target = np.array(labels)
        # target = np.zeros([length])
        return points, target

    def pca_example(self, length):
        points = np.random.uniform(0, 1, [length, 2])
        ind1 = np.where(points[:, 0] < 0.5)[0]
        target = np.zeros([length])
        target[ind1] = 1
        return points, target

    def gaussian_example(self, length, clusters):
        centers = np.random.uniform(0, 1, [clusters, 2])
        per_cl = length // clusters
        Pts = []
        cov = 0.001 * np.eye(2, 2)
        target = np.zeros([length])
        for c in xrange(clusters):
            points = np.random.multivariate_normal(centers[c], cov, per_cl)
            target[c * per_cl: (c + 1) * per_cl] = c
            Pts.append(points)
        points = np.reshape(Pts, [-1, 2])
        rand_perm = np.random.permutation(length)
        points = points[rand_perm]
        target = target[rand_perm]
        return points, target

    def plot_example(self, x, y, clusters, length):
        plt.figure(0)
        plt.clf()
        colors = cm.rainbow(np.linspace(0, 1, clusters))
        for c in xrange(clusters):
            ind = np.where(y == c)[0]
            plt.scatter(x[ind, 0], x[ind, 1], c=colors[c])
        path = '/home/anowak/DynamicProgramming/DP/plots/example.png'
        plt.savefig(path)

    def create(self, clusters=3,  mode='train'):
        if mode == 'train':
            num_examples = self.num_examples_train
        else:
            num_examples = self.num_examples_test
        _, max_length = self.compute_length(clusters)
        x = -1 * np.ones([num_examples, max_length, self.input_size])
        y = 1e6 * np.ones([num_examples, max_length])
        for ex in xrange(num_examples):
            length, max_length = self.compute_length(clusters)
            if self.task == "kmeans":
                # x_ex, y_ex = self.kmeans_example(length, clusters)
                # x_ex, y_ex = self.pca_example(length)
                x_ex, y_ex = self.gaussian_example(length, clusters)
                if ex % 8000 == 7999:
                    print('Created example {}'.format(ex))
                    # self.plot_example(x_ex, y_ex, clusters, length)
            else:
                raise ValueError("task {} not implemented"
                                 .format(self.task))
            x[ex, :length], y[ex, :length] = x_ex, y_ex
        return x, y
