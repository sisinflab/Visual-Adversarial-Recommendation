import numpy as np
import random
import pandas as pd
from PIL import Image
from config.configs import *

class Dataset:
    def __init__(self, args):
        path = '../data/' + args.dataset + '/'
        self.f_feature = path + args.experiment_name + '/features.npy'
        self.model_name = args.model
        self.dataset = args.dataset
        self.f_pos = path + 'trainingset.tsv'
        self.df_train = pd.read_csv(self.f_pos, header=None, sep='\t')
        self.test = path + 'testset.tsv'
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

    def sample_images(self, p):
        u, i_pos = self.pos[p]
        i_neg = i_pos
        while i_neg in self.inter[u] or i_neg in self.coldstart:  # remove the cold start items from negative
            i_neg = random.randrange(self.isz)
        im_pos = Image.open(images_path.format(self.dataset) + str(i_pos) + '.jpg')
        im_neg = Image.open(images_path.format(self.dataset) + str(i_neg) + '.jpg')
        try:
            im_pos.load()
        except ValueError:
            print(f'Image at path {i_pos}.jpg was not loaded correctly!')
        try:
            im_neg.load()
        except ValueError:
            print(f'Image at path {i_neg}.jpg was not loaded correctly!')
        if im_pos.mode != 'RGB':
            im_pos = im_pos.convert(mode='RGB')
        if im_neg.mode != 'RGB':
            im_neg = im_neg.convert(mode='RGB')
        im_pos = (np.array(im_pos.resize((224, 224))) - np.float32(127.5)) / np.float32(127.5)
        im_neg = (np.array(im_neg.resize((224, 224))) - np.float32(127.5)) / np.float32(127.5)
        return u, im_pos, im_neg

    def batch_generator(self):
        self.shuffle()
        sz = len(self.pos) // self.bsz * self.bsz  # in case batch size is not evenly divisible by the dataset size

        for st in range(0, sz, self.bsz):
            if self.model_name == 'VBPR':
                samples = zip(*map(self.sample, range(st, st + self.bsz)))
                yield map(np.array, samples)
            elif self.model_name == 'DVBPR':
                samples = zip(*map(self.sample_images, range(st, st + self.bsz)))
                yield map(np.array, samples)

    def test_generator(self):

        for u in range(0, self.usz):
            pos_items = self.pos_elements[self.pos_elements['u'] == u]['i'].tolist()
            neg_samples = list(set(range(self.isz)).difference(pos_items))
            samples = zip(*[(u, i) for i in neg_samples])
            yield map(np.array, samples)


