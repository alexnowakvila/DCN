import numpy as np
import os


class Generator(object):
    def __init__(
                 self, num_examples_train, num_examples_test,
                 dataset_path, batch_size
                 ):
        self.num_examples_train = num_examples_train
        self.num_examples_test = num_examples_test
        self.batch_size = batch_size
        self.dataset_path = dataset_path
        self.task = 'sort'
        scales_train = [3]
        scales_test = [2, 3, 4]
        self.scales = {'train': scales_train, 'test': scales_test}
        self.data = {'train': {}, 'test': {}}

    def load_dataset(self):
        for mode in ['train', 'test']:
            for sc in self.scales[mode]:
                path = os.path.join(self.dataset_path, mode + str(sc))
                if os.path.exists(path + '.npz'):
                    print('Reading {} dataset for {} scales'
                          .format(mode, sc))
                    npz = np.load(path + '.npz')
                    self.data[mode][sc] = {'x': npz['x'], 'y': npz['y']}
                else:
                    x, y = self.create(scales=sc, mode=mode)
                    self.data[mode][sc] = {'x': x, 'y': y}
                    # save
                    np.savez(path, x=x, y=y)
                    print('Created {} dataset for {} scales'
                          .format(mode, sc))

    def get_batch(self, batch=0, scales=3, mode="train"):
        bs = self.batch_size
        batch_x = self.data[mode][scales]['x'][batch * bs: (batch + 1) * bs]
        batch_y = self.data[mode][scales]['y'][batch * bs: (batch + 1) * bs]
        return batch_x, batch_y

    def compute_length(self, scales):
        length = np.random.randint(2 ** scales, 2 ** scales + 1)
        max_length = 2 ** scales
        return length, max_length

    def build_phi(self, e):
        e1 = np.reshape(e, [-1, 1])
        e2 = np.reshape(e, [1, -1])
        return (e1 == e2).astype(float)

    def split_recursively(self, b, ind, scale, scales):
        if scale == scales:
            return b
        else:
            l = len(ind)
            ind1 = ind[:(l // 2)]
            ind2 = ind[(l // 2):]
            b[ind1, scale] = 0
            b = self.split_recursively(b, ind1, scale + 1, scales)
            b[ind2, scale] = 1
            b = self.split_recursively(b, ind2, scale + 1, scales)
            return b

    def mergesort_split(self, lengths, max_length, scales):
        batch_size = lengths.shape[0]
        Phi = [np.zeros([batch_size, max_length, max_length])
               for scale in xrange(scales)]
        B = [np.zeros([batch_size, max_length]) for scale in xrange(scales)]
        E = np.zeros([batch_size, max_length])
        for batch in xrange(batch_size):
            length = lengths[batch]
            b = np.zeros([length, scales])
            ind = range(length)
            b = self.split_recursively(b, ind, 0, scales)
            e = np.zeros([length])
            for scale in xrange(scales):
                Phi[scale][batch, :length, :length] = self.build_phi(e)
                e = 2 * e + b[:, scale]
                B[scale][batch, :length] = b[:, scale]
            E[batch, :length] = e
        return E, B, Phi

    def sort_example(self, length):
        points = np.random.uniform(0, 1, size=[length, 1])
        target = np.argsort(points[:, 0])
        return points, target

    def create(self, scales=3,  mode='train'):
        if mode == 'train':
            num_examples = self.num_examples_train
        else:
            num_examples = self.num_examples_test
        length, max_length = self.compute_length(scales)
        x = -1 * np.ones([num_examples, max_length, 1])
        y = -1 * np.ones([num_examples, max_length])
        for ex in xrange(num_examples):
            length, max_length = self.compute_length(scales)
            if ex % 500000 == 499999:
                    print('Created example {}'.format(ex))
            if self.task == 'sort':
                x_ex, y_ex = self.sort_example(length)
            else:
                raise ValueError("task {} not implemented"
                                 .format(self.task))
            x[ex, :length], y[ex, :length] = x_ex, y_ex
        return x, y
