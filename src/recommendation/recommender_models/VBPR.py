import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf


class VBPR:

    def __init__(self, args, num_users, num_items, num_image_feature):
        self.emb_K = args.emb1_K
        self.lr = eval(args.lr)
        self.slr = args.lr

        self.regs = args.regs
        regs = eval(self.regs)
        self.l1 = regs[0]
        self.l2 = regs[1]
        self.l3 = regs[2]
        self.lmd = args.lmd
        self.adv = args.adv
        self.adv_type = args.adv_type
        self.epsilon = args.epsilon
        self.num_users = num_users
        self.num_items = num_items
        self.num_image_feature = num_image_feature
        self.watch = []
        self.build_graph()

    def _create_placeholders(self):
        with tf.name_scope("input_data"):
            self.user_input = tf.compat.v1.placeholder(tf.int32, shape=[None], name="user_input")
            self.pos_input = tf.compat.v1.placeholder(tf.int32, shape=[None], name="pos_input")
            self.neg_input = tf.compat.v1.placeholder(tf.int32, shape=[None], name="neg_input")

            self.init_image = tf.compat.v1.placeholder(tf.float32, shape=[self.num_items, self.num_image_feature],
                                                       name="pos_image")
            self.init_emb_P = tf.compat.v1.placeholder(tf.float32, shape=[self.num_users, self.emb_K],
                                                       name="init_emb_P")
            self.init_emb_Q = tf.compat.v1.placeholder(tf.float32, shape=[self.num_items, self.emb_K],
                                                       name="init_emb_Q")

    def _create_variables(self):
        with tf.name_scope("embedding"):
            self.emb_P = tf.Variable(
                tf.random.truncated_normal(shape=[self.num_users, self.emb_K], mean=0.0, stddev=0.01),
                name='emb_P', dtype=tf.float32)  # (users, embedding_size)
            self.emb_Q = tf.Variable(
                tf.random.truncated_normal(shape=[self.num_items, self.emb_K], mean=0.0, stddev=0.01),
                name='emb_Q', dtype=tf.float32)  # (items, embedding_size)

        with tf.name_scope("feature"):
            self.image_feature = tf.Variable(
                tf.random.truncated_normal(shape=[self.num_items, self.num_image_feature], mean=0.0, stddev=0.01),
                name='image_feature', dtype=tf.float32, trainable=False)  # (items, embedding_size)

        with tf.name_scope("init_op"):
            self.assign_image = tf.compat.v1.assign(self.image_feature, self.init_image)
            self.assign_P = tf.compat.v1.assign(self.emb_P, self.init_emb_P)
            self.assign_Q = tf.compat.v1.assign(self.emb_Q, self.init_emb_Q)

        with tf.name_scope("image_transfer"):
            self.phi = tf.Variable(
                tf.random.truncated_normal(shape=[self.num_image_feature, self.emb_K], mean=0.0, stddev=0.01),
                name='phi', dtype=tf.float32)

    def _create_inference(self, user_input, item_input, adv=False):
        with tf.name_scope("inference"):
            self.emb_p = tf.nn.embedding_lookup(self.emb_P, user_input)
            self.emb1_q = tf.nn.embedding_lookup(self.emb_Q, item_input)
            image_input = tf.nn.embedding_lookup(self.image_feature, item_input)
            self.emb2_q = tf.matmul(image_input, self.phi)
            if adv:
                gd = tf.nn.embedding_lookup(self.delta, item_input)
                self.d = self.epsilon * tf.nn.l2_normalize(gd, 1)
                self.watch.append(self.d)
                self.emb2_q = self.emb2_q + tf.matmul(self.d, self.phi)

            self.emb_q = self.emb1_q + self.emb2_q

            return tf.reduce_sum(self.emb_p * self.emb_q, 1), self.emb_p, self.emb_q, image_input

    def _prediction(self):
        with tf.name_scope("prediction"):
            self.emb2_Q = tf.matmul(self.image_feature, self.phi)
            if self.adv:
                self.d = self.epsilon * tf.nn.l2_normalize(self.delta, 1)
                self.watch.append(self.d)
                self.emb2_Q = self.emb2_Q + tf.matmul(self.d, self.phi)

            self.temp_emb_Q = self.emb_Q + self.emb2_Q

            self.predictions = tf.matmul(self.emb_P * -1, self.temp_emb_Q, transpose_b=True)

    def _create_loss(self):
        self.pos_pred, emb_p, emb_pos_q, self.emb_pos_feature = self._create_inference(self.user_input, self.pos_input)
        self.neg_pred, _, emb_neg_q, _ = self._create_inference(self.user_input, self.neg_input)

        self.result = self.pos_pred - self.neg_pred
        self.loss = tf.reduce_sum(tf.nn.softplus(-self.result))

        self.adv_loss = 0

        if self.adv:
            if self.adv_type == 'rand':
                self.delta = tf.random.truncated_normal(shape=self.image_feature.shape, mean=0.0, stddev=0.01)
            else:
                self.delta = tf.gradients(self.loss, [self.image_feature])[0]
            self.delta = tf.stop_gradient(self.delta)

            self.pos_pred_adv, _, _, _ = self._create_inference(self.user_input, self.pos_input, adv=True)
            self.neg_pred_adv, _, _, _ = self._create_inference(self.user_input, self.neg_input, adv=True)

            result_adv = self.pos_pred_adv - self.neg_pred_adv
            self.adv_loss = tf.reduce_sum(tf.nn.softplus(-result_adv))

        self.opt_loss = self.loss + self.lmd * self.adv_loss \
                        + self.l1 * (tf.reduce_sum(self.emb_p)) \
                        + self.l2 * (tf.reduce_sum(emb_pos_q) + tf.reduce_sum(emb_neg_q)) \
                        + self.l3 * (tf.reduce_sum(self.phi))

    def _create_optimizer(self):
        with tf.name_scope("optimizer"):

            vlist = [self.emb_P]
            vlist2 = [self.emb_Q]
            vlist3 = [self.phi]

            if isinstance(self.lr, list):
                lr = self.lr
            else:
                lr = [self.lr, self.lr, self.lr]

            opt = tf.compat.v1.train.AdagradOptimizer(lr[0])
            opt2 = tf.compat.v1.train.AdagradOptimizer(lr[1])
            opt3 = tf.compat.v1.train.AdagradOptimizer(lr[2])
            grads_all = tf.gradients(self.opt_loss, vlist + vlist2 + vlist3)
            grads = grads_all[0:1]
            grads2 = grads_all[1:2]
            grads3 = grads_all[2:3]
            train_op = opt.apply_gradients(zip(grads, vlist))
            train_op2 = opt2.apply_gradients(zip(grads2, vlist2))
            train_op3 = opt3.apply_gradients(zip(grads3, vlist3))

            self.optimizer = tf.group(train_op, train_op2, train_op3)

    def build_graph(self):
        self._create_placeholders()
        self._create_variables()
        self._create_loss()
        self._create_optimizer()
        self._prediction()

    def get_saver_name(self):
        return "stored_vbpr_k_%d_lr_%s_regs_%s_eps_%f_lmd_%f" % \
               (self.emb_K, self.slr, self.regs, self.epsilon, self.lmd)

