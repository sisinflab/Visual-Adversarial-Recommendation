import logging
import random
import numpy as np
import tensorflow as tf
import concurrent.futures

random.seed(0)
np.random.seed(0)
tf.random.set_random_seed(0)
logging.disable(logging.WARNING)


class ACF:
    def __init__(self, args, dataset):
        self.data = dataset
        self.emb_K = args.emb1_K
        self.regs = args.regs[0]
        self.lr = args.lr[0]
        self.num_users = dataset.usz
        self.num_items = dataset.isz

        self.layers_component = args.layers_component
        self.layers_item = args.layers_item

        self.feature_shape = dataset.emb_image_shape
        self.initializer_attentive = tf.glorot_normal_initializer()

        self.Gu, self.Gi, self.Pi = self._initialize_variables()
        self.component_weights, self.item_weights = self._build_attention_weights()

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)

    def _initialize_variables(self):
        with tf.name_scope("embeddings"):
            return tf.Variable(tf.random_normal(shape=[self.num_users, self.emb_K], stddev=0.01),
                               name='Gu', dtype=tf.float32), \
                tf.Variable(tf.random_normal(shape=[self.num_items, self.emb_K], stddev=0.01),
                            name='Gi', dtype=tf.float32), \
                tf.Variable(tf.random_normal(shape=[self.num_items, self.emb_K], stddev=0.01),
                            name='Pi', dtype=tf.float32) \


    def _build_attention_weights(self):
        component_dict = dict()
        items_dict = dict()

        for c in range(len(self.layers_component)):
            # the inner layer has all components
            if c == 0:
                component_dict['W_{}_u'.format(c)] = tf.Variable(
                    self.initializer_attentive(shape=[self.emb_K, self.layers_component[c]]),
                    name='W_{}_u'.format(c),
                    dtype=tf.float32
                )
                component_dict['W_{}_i'.format(c)] = tf.Variable(
                    self.initializer_attentive(shape=[self.feature_shape[-1], self.layers_component[c]]),
                    name='W_{}_i'.format(c),
                    dtype=tf.float32
                )
                component_dict['b_{}'.format(c)] = tf.Variable(
                    self.initializer_attentive(shape=[self.layers_component[c]]),
                    name='b_{}'.format(c),
                    dtype=tf.float32
                )
            else:
                component_dict['W_{}'.format(c)] = tf.Variable(
                    self.initializer_attentive(shape=[self.layers_component[c], self.layers_component[c - 1]]),
                    name='W_{}_u'.format(c),
                    dtype=tf.float32
                )
                component_dict['b_{}'.format(c)] = tf.Variable(
                    self.initializer_attentive(shape=[self.layers_component[c]]),
                    name='b_{}'.format(c),
                    dtype=tf.float32
                )

        for i in range(len(self.layers_item)):
            # the inner layer has all components
            if i == 0:
                items_dict['W_{}_u'.format(i)] = tf.Variable(
                    self.initializer_attentive(shape=[self.emb_K, self.layers_item[i]]),
                    name='W_{}_u'.format(i),
                    dtype=tf.float32
                )
                items_dict['W_{}_iv'.format(i)] = tf.Variable(
                    self.initializer_attentive(shape=[self.emb_K, self.layers_item[i]]),
                    name='W_{}_iv'.format(i),
                    dtype=tf.float32
                )
                items_dict['W_{}_ip'.format(i)] = tf.Variable(
                    self.initializer_attentive(shape=[self.emb_K, self.layers_item[i]]),
                    name='W_{}_ip'.format(i),
                    dtype=tf.float32
                )
                items_dict['W_{}_ix'.format(i)] = tf.Variable(
                    self.initializer_attentive(shape=[self.feature_shape[-1], self.layers_item[i]]),
                    name='W_{}_ix'.format(i),
                    dtype=tf.float32
                )
                items_dict['b_{}'.format(i)] = tf.Variable(
                    self.initializer_attentive(shape=[self.layers_item[i]]),
                    name='b_{}'.format(i),
                    dtype=tf.float32
                )
            else:
                items_dict['W_{}'.format(i)] = tf.Variable(
                    self.initializer_attentive(shape=[self.layers_item[i], self.layers_item[i - 1]]),
                    name='W_{}_u'.format(i),
                    dtype=tf.float32
                )
                items_dict['b_{}'.format(i)] = tf.Variable(
                    self.initializer_attentive(shape=[self.layers_item[i]]),
                    name='b_{}'.format(i),
                    dtype=tf.float32
                )
        return component_dict, items_dict

    def _calculate_beta_alpha(self, i_p):
        # calculate beta
        u, list_of_pos = i_p['u'], i_p['u_pos']
        g_u = tf.expand_dims(tf.nn.embedding_lookup(self.Gu, u), axis=1)

        f_i_np = np.empty(shape=(len(list_of_pos), self.feature_shape[1] * self.feature_shape[2],
                                 self.feature_shape[3]), dtype=np.float32)

        for index, p in enumerate(list_of_pos):
            f_i_np[index] = np.load(self.data.f_feature + str(p)
                                    + '.npy').reshape((self.feature_shape[1] * self.feature_shape[2],
                                                       self.feature_shape[3]))

        f_i = tf.Variable(f_i_np, dtype=tf.float32)
        del f_i_np

        b_i_l = tf.squeeze(tf.matmul(self.component_weights['W_{}_u'.format(0)], g_u, transpose_a=True)) + \
                tf.tensordot(f_i, self.component_weights['W_{}_i'.format(0)], axes=[[2], [0]]) + \
                self.component_weights['b_{}'.format(0)]
        b_i_l = tf.nn.relu(b_i_l)
        for c in range(1, len(self.layers_component)):
            b_i_l = tf.tensordot(b_i_l, self.component_weights['W_{}'.format(c)], axes=[[2], [1]]) + \
                    self.component_weights['b_{}'.format(c)]

        b_i_l = tf.nn.softmax(tf.squeeze(b_i_l, -1), axis=1)
        all_x_l = tf.reduce_sum(tf.multiply(tf.expand_dims(b_i_l, axis=2), f_i), axis=1)

        # calculate alpha
        g_i = tf.nn.embedding_lookup(self.Gi, list_of_pos)
        p_i = tf.nn.embedding_lookup(self.Pi, list_of_pos)
        a_i_l = tf.squeeze(tf.matmul(self.item_weights['W_{}_u'.format(0)], g_u, transpose_a=True)) + \
                tf.matmul(g_i, self.item_weights['W_{}_iv'.format(0)]) + \
                tf.matmul(p_i, self.item_weights['W_{}_ip'.format(0)]) + \
                tf.matmul(all_x_l, self.item_weights['W_{}_ix'.format(0)]) + \
                self.item_weights['b_{}'.format(0)]
        a_i_l = tf.nn.relu(a_i_l)
        for c in range(1, len(self.layers_item)):
            a_i_l = tf.matmul(a_i_l, self.item_weights['W_{}'.format(c)], transpose_b=True) + \
                    self.item_weights['b_{}'.format(c)]
        a_i_l = tf.nn.softmax(tf.reshape(a_i_l, [-1]))

        all_a_i_l = tf.reduce_sum(tf.multiply(tf.expand_dims(a_i_l, axis=1), p_i), axis=0)
        g_u_p = tf.squeeze(g_u) + all_a_i_l

        return g_u_p

    def _forward(self, user, item):
        gamma_u = tf.squeeze(tf.nn.embedding_lookup(self.Gu, user))
        gamma_i = tf.squeeze(tf.nn.embedding_lookup(self.Gi, item))
        p_i = tf.squeeze(tf.nn.embedding_lookup(self.Pi, item))

        all_pos_u = [{'u': i.numpy(), 'u_pos': list(self.data.inter[i.numpy()])} for i in user]

        with concurrent.futures.ThreadPoolExecutor() as executor:
            output = executor.map(self._calculate_beta_alpha, all_pos_u)

        gamma_u_p = tf.Variable(np.asarray(list(output)))

        xui = tf.reduce_sum(gamma_u_p * gamma_i, 1)
        return xui, gamma_u, gamma_i, p_i

    def _loss(self, user, pos, neg):
        xu_pos, gamma_u, gamma_pos, p_i_pos = self._forward(user, pos)
        xu_neg, _, gamma_neg, p_i_neg = self._forward(user, neg)

        result = tf.clip_by_value(xu_pos - xu_neg, -80.0, 1e8)
        loss = tf.reduce_sum(tf.nn.softplus(-result))

        opt_loss = loss + self.regs * tf.reduce_sum([tf.nn.l2_loss(gamma_u),
                                                     tf.nn.l2_loss(gamma_pos),
                                                     tf.nn.l2_loss(gamma_neg),
                                                     tf.nn.l2_loss(p_i_pos),
                                                     tf.nn.l2_loss(p_i_neg),
                                                     *[tf.nn.l2_loss(value)
                                                       for _, value in self.component_weights.items()],
                                                     *[tf.nn.l2_loss(value)
                                                       for _, value in self.item_weights.items()]])

        return opt_loss

    def train_step(self, inputs):
        with tf.GradientTape() as t:
            user, pos, neg = inputs
            opt_loss = self._loss(user, pos, neg)

        params = [self.Gu,
                  self.Gi,
                  self.Pi,
                  *[value for _, value in self.component_weights.items()],
                  *[value for _, value in self.item_weights.items()]]

        grads = t.gradient(opt_loss, params)
        self.optimizer.apply_gradients(zip(grads, params))

        return opt_loss.numpy()

    def prediction_all(self):
        all_pos_u = [{'u': i, 'u_pos': self.data.inter[i]} for i in range(self.data.usz)]

        with concurrent.futures.ThreadPoolExecutor() as executor:
            output = executor.map(self._calculate_beta_alpha, all_pos_u)

        gu_p = tf.Variable(np.asarray(list(output)))
        return tf.matmul(gu_p, self.Gi, transpose_b=True)

    def get_model_params(self):
        return {
            'Gu': self.Gu.numpy(),
            'Gi': self.Gi.numpy(),
            'Pi': self.Pi.numpy(),
            'Component Weights': self.component_weights,
            'Item Weights': self.item_weights
        }, self.optimizer

    def restore(self, params):
        self.Gu.assign(params[0].get('Gu'))
        self.Gi.assign(params[0].get('Gi'))
        self.Pi.assign(params[0].get('Pi'))
        for key, value in self.component_weights:
            value.assign(params[0].get('Component Weights')[key])
        for key, value in self.item_weights:
            value.assign(params[0].get('Item Weights')[key])
        self.optimizer.assign(params[1])
