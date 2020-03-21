import math
import time
import utils.read as read
import utils.write as write

import numpy as np
import tensorflow as tf
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = ""

from recommendation.recommender_models.VBPR import VBPR
from recommendation.recommender_models.AMR import AMR
from recommendation.recommender_dataset.Dataset import Dataset


class Solver:
    def __init__(self, args):
        self.dataset = Dataset(args)
        self.dataset_name = args.dataset
        self.experiment_name = args.experiment_name
        self.adv = args.adv
        if self.adv:
            self.model = AMR(args, self.dataset.usz, self.dataset.isz, self.dataset.fsz)
        else:
            self.model = VBPR(args, self.dataset.usz, self.dataset.isz, self.dataset.fsz)
        self.epoch = args.epoch
        self.verbose = args.verbose

        self.sess = tf.compat.v1.Session()
        self.sess.run(tf.compat.v1.global_variables_initializer())
        self.saver = tf.compat.v1.train.Saver(tf.compat.v1.trainable_variables(), max_to_keep=0)
        self.sess.run(self.model.assign_image, feed_dict={self.model.init_image: self.dataset.emb_image})

        self.tp_k_predictions = args.tp_k_predictions
        self.weight_dir = '../' + args.weight_dir + '/'
        self.result_dir = '../' + args.result_dir + '/'

        if self.experiment_name == 'original_images':
            self.attack_type = 'original_images'
            self.attacked_categories = ''
            self.eps_cnn = ''
            self.iteration_attack_type = ''
            self.norm = ''
        else:
            self.attack_type = self.experiment_name.split('_')[0]
            self.attacked_categories = '_' + self.experiment_name.split('_')[1] + '_' + self.experiment_name.split('_')[
                2]
            self.eps_cnn = '_' + self.experiment_name.split('_')[3]
            self.iteration_attack_type = '_' + self.experiment_name.split('_')[4]
            self.norm = '_' + self.experiment_name.split('_')[5]

        self.experiment_name = '{0}/{1}'.format(self.dataset_name, self.experiment_name)

        if self.adv:
            self.load()

    def one_epoch(self):
        generator = self.dataset.batch_generator()
        api = [self.model.user_input, self.model.pos_input, self.model.neg_input]
        while True:
            try:
                feed_dict = dict(zip(api, next(generator)))
                self.sess.run([self.model.optimizer], feed_dict=feed_dict)
            except StopIteration:
                break

    def train(self):
        start_epoch = 0
        if self.adv:
            start_epoch = self.epoch // 2

        for i in range(start_epoch + 1, self.epoch + 1):
            start = time.time()
            self.one_epoch()
            if i % self.verbose == 0:
                self.save(i)
            print('Epoch {0}/{1} in {2} secs.'.format(i, self.epoch, time.time() - start))


        self.store_predictions(i)
        self.save(i)

    def evaluate_rec_metrics(self, para):
        r, K = para
        hr = 1 if r < K else 0
        if hr:
            ndcg = math.log(2) / math.log(r + 2)
        else:
            ndcg = 0
        return hr, ndcg

    def original_test(self, message):
        st = time.time()
        generator = self.dataset.test_generator()
        api = [self.model.user_input, self.model.pos_input]
        d = []
        i = 0
        start = time.time()
        while True:
            try:
                feed_dict = dict(zip(api, next(generator)))
                preds = self.sess.run(self.model.pos_pred, feed_dict=feed_dict)

                rank = np.sum(preds[1:] >= preds[0])
                d.append(rank)

                i += 1
                if i % 1000 == 0:
                    print("Tested {0}/{1} in {2}".format(i, self.dataset.usz, time.time() - start))
                    start = time.time()

            except Exception as e:
                # print type(e), e.message
                break
        score5 = np.mean([ele for ele in map(self.evaluate_rec_metrics, zip(d, [5] * len(d)))], 0)
        score10 = np.mean([ele for ele in map(self.evaluate_rec_metrics, zip(d, [10] * len(d)))], 0)
        score20 = np.mean([ele for ele in map(self.evaluate_rec_metrics, zip(d, [20] * len(d)))], 0)

        print(message, score5, score10, score20)
        print('evaluation cost', time.time() - st)

    def store_predictions(self, epoch):
        # We multiply the users embeddings by -1 to have the np sorting operation in the correct order

        print('Start Store Predictions at epoch {0}'.format(epoch))
        start = time.time()
        # predictions = self.sess.run(self.model.predictions)
        emb_P = self.sess.run(self.model.emb_P)*-1
        temp_emb_Q = self.sess.run(self.model.temp_emb_Q)
        predictions = np.matmul(emb_P, temp_emb_Q.transpose())
        predictions = predictions.argsort(axis=1)
        predictions = [predictions[i][:self.tp_k_predictions] for i in range(predictions.shape[0])]
        prediction_name = self.result_dir + self.experiment_name + 'top{0}_predictions_epoch{1}'.format(
            self.tp_k_predictions, epoch)
        if self.adv:
            prediction_name = prediction_name + '_AMR'

        write.save_obj(predictions, prediction_name)

        print('End Store Predictions {0}'.format(time.time() - start))

    def load(self):
        try:
            params = np.load(self.weight_dir + self.experiment_name + 'step{0}.npy'.format(self.epoch//2), allow_pickle=True)
            self.sess.run([self.model.assign_P, self.model.assign_Q, self.model.phi.assign(params[2])],
                          {self.model.init_emb_P: params[0], self.model.init_emb_Q: params[1]})
            print('Load parameters from {0}'.format(self.weight_dir + self.experiment_name + 'step2000.npy'))
        except Exception as ex:
            print('Start new model from scratch')

    def save(self, step):
        params = self.sess.run(tf.compat.v1.trainable_variables())
        store_model_path = self.weight_dir + self.experiment_name + 'step{0}'.format(step)
        if self.adv:
            store_model_path = store_model_path + '_AMR'

        np.save(store_model_path + '.npy', params)
