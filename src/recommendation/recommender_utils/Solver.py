import math
import time
import utils.read
import utils.write

import numpy as np
import tensorflow as tf

from recommendation.recommender_models import VBPR
from recommendation.recommender_dataset import Dataset


class Solver:
    def __init__(self, args):
        self.dataset = Dataset(args)
        self.model = VBPR(args, self.dataset.usz, self.dataset.isz, self.dataset.fsz)
        self.epoch = args.epoch
        self.verbose = args.verbose
        self.adv = args.adv
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=0)
        self.sess.run(self.model.assign_image, feed_dict={self.model.init_image: self.dataset.emb_image})
        self.tp_k_predictions = args.tp_k_predictions
        self.weight_dir = args.weight_dir + '/'

        self.result_dir = args.result_dir + '/'
        self.attack_type = args.attack_type
        self.iteration_attack_type = args.iteration_attack_type
        self.attacked_categories = args.attacked_categories
        self.experiment_dir = args.experiment_dir
        self.eps_cnn = args.eps_cnn

        self.experiment_name = '{0}/{1}_{2}_eps{3}_it{4}'.format(self.dataset, self.attack_type, self.attacked_categories, self.eps_cnn, self.iteration_attack_type)

        self.load()

    def one_epoch(self):
        generator = self.dataset.batch_generator()
        api = [self.model.user_input, self.model.pos_input, self.model.neg_input]
        while True:
            feed_dict = dict(zip(api, next(generator)))
            self.sess.run([self.model.optimizer], feed_dict=feed_dict)

    def train(self):

        for i in range(0, self.epoch ):
            start = time.time()
            if i % self.verbose == 0:
                # self.test('epoch %d' % i)
                self.store_predictions('epoch %d' % i)
                self.save(i)
            self.one_epoch()
            print('Epoch {0}/{1} in {2} secs.'.format(i, self.epoch, time.time() - start))
        # self.save(i)

    @staticmethod
    def _score(para):
        r, K = para
        hr = r < K
        if hr:
            ndcg = math.log(2) / math.log(r + 2)
        else:
            ndcg = 0
        return hr, ndcg

    def test(self, message):
        results = {}
        generator = self.dataset.test_generator()
        api = [self.model.user_input, self.model.pos_input]
        d = []
        i = 0

        print('Start Test')
        start = time.time()
        while True:
            # For each user
            try:
                feeds, positive_items, user_id = generator.next()
                feed_dict = dict(zip(api, feeds))
                # In pred we have also the 'already rated items'.
                preds = self.sess.run(self.model.pos_pred, feed_dict=feed_dict)
                rank = np.sum(preds[1:] >= preds[0])
                d.append(rank)

                i += 1
                if i % 100 == 0:
                    print("Tested {0}/{1} in {2}".format(i, self.dataset.usz, time.time() - start))
                    start = time.time()

            except Exception as e:
                print(type(e), e.message)
                break

        score5 = np.mean(map(self._score, zip(d, [5] * len(d))), 0)

        write.save_obj(results, self.result_dir + self.experiment_name)
        print('Test Results stored')

    def store_predictions(self, message):
        # We multiply the users embeddings by -1 to have the np sorting operation in the correct order
        predictions = self.sess.run(self.model.predictions)
        predictions = predictions.argsort(axis=1)
        predictions = [predictions[i][:self.tp_k_predictions] for i in range(predictions.shape[0])]
        write.save_obj(predictions, self.result_dir + self.experiment_name + '_top{0}_predictions'.format(self.tp_k_predictions))

    def load(self):
        try:
            params = np.load(self.weight_dir + 'best-vbpr.npy', allow_pickle=True)
            self.sess.run([self.model.assign_P, self.model.assign_Q, self.model.phi.assign(params[2])],
                          {self.model.init_emb_P: params[0], self.model.init_emb_Q: params[1]})
            print('Load parameters from best-vbpr.npy')
        except:
            print('Start new model from scratch')

    def save(self, step):
        params = self.sess.run(tf.trainable_variables())
        store_model_path = self.weight_dir + self.experiment_name + '/{0}_step{1}.npy'.format(self.model.get_saver_name(), step)
        np.save(store_model_path, params)
