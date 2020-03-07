import numpy as np
import random
import pandas as pd


class Dataset:
    def __init__(self, args):
        path = '../data/' + args.dataset + '/'
        self.f_feature = path + args.experiment_name + '/features.npy'
        self.f_train = path + 'train.txt'
        self.bsz = args.batch_size
        self.emb_image = np.load(self.f_feature)
        self.fsz = self.emb_image.shape[1]
        self.train = np.loadtxt(self.f_train, dtype=np.int)
        self.usz, self.isz = np.max(self.train, 0) + 1
        self.train_elements = pd.read_csv(self.f_train, sep='\t', header=None)
        self.train_elements.columns = ['u', 'i']
        self.train_elements.u = self.train_elements.u.astype(int)
        self.train_elements.i = self.train_elements.i.astype(int)
        self.emb_image = self.emb_image / np.max(np.abs(self.emb_image))
        # self.coldstart = set(self.neg[:,0].tolist()) - set(self.train[:,1].tolist())
        self.coldstart = set(range(0, self.isz)) - set(self.train[:, 1].tolist())
        self.train = list(self.train)

        self.inter = {}
        for u, i in self.train:
            if u not in self.inter:
                self.inter[u] = set([])
            self.inter[u].add(i)

    def shuffle(self):
        random.shuffle(self.train)

    def sample(self, p):
        u, i = self.train[p]
        i_neg = i
        while i_neg in self.inter[u] or i_neg in self.coldstart:  # remove the cold start items from negative samples
            i_neg = random.randrange(self.isz)
        return u, i, i_neg

    def batch_generator(self):
        self.shuffle()
        sz = len(self.train)//self.bsz*self.bsz

        for st in range(0, sz, self.bsz):
            samples = zip(*map(self.sample, range(st, st + self.bsz)))
            yield map(np.array, samples)

    def test_generator(self):

        for u in range(0, self.usz):
            pos_items = self.train_elements[self.train_elements['u'] == u]['i'].tolist()
            neg_samples = list(set(range(self.isz)).difference(pos_items))
            samples = zip(*[(u, i) for i in neg_samples])
            yield map(np.array, samples)

