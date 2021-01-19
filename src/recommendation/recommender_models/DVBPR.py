from recommendation.cnn.cnn import *
from config.configs import *
from PIL import Image
import tensorflow as tf
import logging
import random
import numpy as np
import os

random.seed(0)
np.random.seed(0)
tf.compat.v1.random.set_random_seed(0)
logging.disable(logging.WARNING)


class DVBPR:
    def __init__(self, args, num_users, num_items):
        self.dataset = args.dataset
        self.experiment_name = '{0}/{1}'.format(self.dataset, args.experiment_name)
        self.emb_K = args.emb1_K
        self.regs = args.regs
        self.lambda1 = self.regs[0]
        self.lambda2 = self.regs[1]
        self.num_users = num_users
        self.num_items = num_items
        self.lr = args.lr[0]
        self._initialize_variables()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)

    def _initialize_variables(self):
        with tf.name_scope("embeddings"):
            self.Tu = tf.Variable(tf.random_uniform(shape=[self.num_users, self.emb_K],
                                                    minval=0, maxval=1) / 100, name='Tu', dtype=tf.float32)
        with tf.name_scope("cnn"):
            self.cnn = CNN(self.emb_K)
        self.Phi = np.empty(shape=[self.num_items, self.emb_K], dtype=np.float32)

    def _forward_cnn(self, item):
        return self.cnn(inputs=item, training=False)

    def _forward(self, user, item):
        cnn_output = self.cnn(inputs=item, training=True)
        theta_u = tf.nn.embedding_lookup(self.Tu, user)

        xui = tf.matmul(theta_u, cnn_output, transpose_b=True)

        return xui, theta_u, cnn_output

    def prediction_all(self):
        # load all images and calculate phi for each of them
        # assign phi to Phi to get the overall Phi vector
        # calculate the prediction for all users-items
        images_list = os.listdir(images_DVBPR_path.format(self.experiment_name))
        images_list.sort(key=lambda x: int(x.split(".")[0]))
        for index, item in enumerate(images_list):
            im = Image.open(images_path.format(self.dataset) + item)

            try:
                im.load()
            except ValueError:
                print(f'Image at path {images_path.format(self.dataset) + item} was not loaded correctly!')

            if im.mode != 'RGB':
                im = im.convert(mode='RGB')

            im = np.reshape((np.array(im.resize((224, 224))) - np.float32(127.5)) / np.float32(127.5), (1, 224, 224, 3))
            phi = self._forward_cnn(im)
            self.Phi[index, :] = phi
        return tf.matmul(self.Tu, tf.Variable(self.Phi), transpose_b=True)

    def _loss(self, user, pos, neg):
        xu_pos, theta_u, phi_pos = self._forward(user, pos)
        xu_neg, theta_u, phi_neg = self._forward(user, neg)

        difference = tf.clip_by_value(xu_pos - xu_neg, -80.0, 1e8)
        loss = tf.reduce_sum(tf.nn.softplus(-difference))

        opt_loss = loss \
                    + self.lambda1 * tf.nn.l2_loss(theta_u) \
                    + self.lambda2 * tf.reduce_sum([tf.nn.l2_loss(layer)
                                                    for layer in self.cnn.trainable_variables
                                                    if 'bias' not in layer.name])
        return opt_loss

    def train_step(self, inputs):
        with tf.GradientTape() as tape:
            user, pos, im_pos, neg, im_neg = inputs
            opt_loss = self._loss(user, im_pos, im_neg)

        params = [self.Tu,
                  *self.cnn.trainable_variables]

        grads = tape.gradient(opt_loss, params)
        self.optimizer.apply_gradients(zip(grads, params))

        return opt_loss.numpy()

    def get_model_params(self):
        return {
            'Tu': self.Tu.numpy(),
            'CNN': [layer.numpy() for layer in self.cnn.trainable_variables]
        }, self.optimizer

    def restore(self, params):
        self.Tu.assign(params[0].get('Tu'))
        for i, layer in enumerate(self.cnn.trainable_variables):
            layer.assign(params[0].get('CNN')[i])
        self.optimizer.assign(params[1])
