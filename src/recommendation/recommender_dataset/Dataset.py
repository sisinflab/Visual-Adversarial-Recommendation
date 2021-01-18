import numpy as np
import random
import pandas as pd
import tensorflow as tf
from PIL import Image
from config.configs import *


class Dataset:
    def __init__(self, args):
        path = '../data/' + args.dataset + '/'
        self.epochs = args.epoch
        self.model_name = args.model
        self.dataset = args.dataset
        self.f_pos = path + 'trainingset.tsv'
        self.df_train = pd.read_csv(self.f_pos, header=None, sep='\t')
        self.test = path + 'testset.tsv'
        self.bsz = args.batch_size

        if self.model_name not in ['DVBPR', 'ACF']:
            self.f_feature = path + args.experiment_name + '/features.npy'
            self.emb_image = np.load(self.f_feature)
            self.emb_image = self.emb_image / np.max(np.abs(self.emb_image))
            self.fsz = self.emb_image.shape[1]
        elif self.model_name == 'ACF':
            self.f_feature = '../data/' + args.dataset + '/' + args.experiment_name + '/features/'
            emb_image = np.load(self.f_feature + '0.npy')
            self.emb_image_shape = emb_image.shape
        else:
            raise NotImplemented('Model not implemented yet!')

        self.pos = np.loadtxt(self.f_pos, dtype=np.int)
        self.usz, self.isz = np.max(self.pos, 0) + 1
        self.pos_elements = pd.read_csv(self.f_pos, sep='\t', header=None)
        self.pos_elements.columns = ['u', 'i']
        self.pos_elements.u = self.pos_elements.u.astype(int)
        self.pos_elements.i = self.pos_elements.i.astype(int)
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

    def read_images_triple(self, user, pos, neg):
        im_pos = Image.open(images_path.format(self.dataset) + str(pos.numpy()) + '.jpg')
        im_neg = Image.open(images_path.format(self.dataset) + str(neg.numpy()) + '.jpg')

        try:
            im_pos.load()
        except ValueError:
            print(f'Image at path {pos.numpy()}.jpg was not loaded correctly!')

        try:
            im_neg.load()
        except ValueError:
            print(f'Image at path {neg.numpy()}.jpg was not loaded correctly!')

        if im_pos.mode != 'RGB':
            im_pos = im_pos.convert(mode='RGB')
        if im_neg.mode != 'RGB':
            im_neg = im_neg.convert(mode='RGB')

        im_pos = (np.array(im_pos.resize((224, 224))) - np.float32(127.5)) / np.float32(127.5)
        im_neg = (np.array(im_neg.resize((224, 224))) - np.float32(127.5)) / np.float32(127.5)
        return user.numpy(), pos.numpy(), im_pos, neg.numpy(), im_neg

    def all_triple_batches(self):
        r_int = np.random.randint
        user_input, pos_input, neg_input = [], [], []
        users_list = list(range(self.usz))
        counter_inter = 1

        actual_inter = (len(self.pos_elements) // self.bsz) * \
                       self.bsz * self.epochs

        for ep in range(self.epochs):
            shuffled_users_list = users_list[:]
            random.shuffle(shuffled_users_list)
            for ab in range(self.usz):
                u = shuffled_users_list[ab]
                uis = self.inter[u]

                for i in uis:
                    j = r_int(self.isz)
                    while j in uis:
                        j = r_int(self.isz)

                    user_input.append(np.array(u))
                    pos_input.append(np.array(i))
                    neg_input.append(np.array(j))

                    if counter_inter == actual_inter:
                        return user_input, pos_input, neg_input,
                    else:
                        counter_inter += 1

        return user_input, pos_input, neg_input,

    def next_triple_batch(self):
        all_triples = self.all_triple_batches()
        data = tf.data.Dataset.from_tensor_slices(all_triples)
        data = data.batch(batch_size=self.bsz)
        data = data.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        return data

    def next_triple_batch_pipeline(self):
        def load_func(u, p, n):
            b = tf.py_function(
                self.read_images_triple,
                (u, p, n,),
                (np.int32, np.int32, np.float32, np.int32, np.float32)
            )
            return b

        all_triples = self.all_triple_batches()
        data = tf.data.Dataset.from_tensor_slices(all_triples)
        data = data.map(load_func, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        data = data.batch(batch_size=self.bsz)
        data = data.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        return data

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


