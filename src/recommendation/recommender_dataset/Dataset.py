import numpy as np
import random
import pandas as pd


class Dataset:
    def __init__(self, args):
        path = '../data/' + args.dataset + '/'
        self.f_feature = path + args.experiment_name + '/features.npy'
        self.f_pos = path + 'pos.txt'
        self.bsz = args.batch_size
        self.emb_image = np.load(self.f_feature)
        self.fsz = self.emb_image.shape[1]
        self.pos = np.loadtxt(self.f_pos, dtype=np.int)
        self.usz, self.isz = np.max(self.pos, 0) + 1
        self.pos_elements = pd.read_csv(self.f_pos, sep='\t', header=None)
        self.pos_elements.columns = ['u', 'i']
        self.pos_elements.u = self.pos_elements.u.astype(int)
        self.pos_elements.i = self.pos_elements.i.astype(int)
        self.emb_image = self.emb_image / np.max(np.abs(self.emb_image))
        # self.coldstart = set(self.neg[:,0].tolist()) - set(self.pos[:,1].tolist())
        self.coldstart = set(range(0, self.isz)) - set(self.pos[:, 1].tolist())
        self.pos = list(self.pos)

        self.inter = {}
        for u, i in self.pos:
            if u not in self.inter:
                self.inter[u] = set([])
            self.inter[u].add(i)

    def shuffle(self):
        random.shuffle(self.pos)

    def sample(self, p):
        u, i = self.pos[p]
        i_neg = i
        while i_neg in self.inter[u] or i_neg in self.coldstart:  # remove the cold start items from negative samples
            i_neg = random.randrange(self.isz)
        return u, i, i_neg

    def batch_generator(self):
        self.shuffle()
        sz = len(self.pos)//self.bsz*self.bsz

        for st in range(0, sz, self.bsz):
            samples = zip(*map(self.sample, range(st, st + self.bsz)))
            yield map(np.array, samples)

    def test_generator(self):

        for u in range(0, self.usz):
            pos_items = self.pos_elements[self.pos_elements['u'] == u]['i'].tolist()
            neg_samples = list(set(range(self.isz)).difference(pos_items))
            samples = zip(*[(u, i) for i in neg_samples])
            yield map(np.array, samples)

