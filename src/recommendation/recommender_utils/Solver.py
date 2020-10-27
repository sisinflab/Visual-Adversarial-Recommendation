import math
import time
import utils.read as read
import utils.write as write
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import os

from recommendation.recommender_models.VBPR import VBPR
from recommendation.recommender_models.DVBPR import DVBPR
from recommendation.recommender_models.AMR import AMR
from recommendation.recommender_dataset.Dataset import Dataset
from config.configs import *


class Solver:
    def __init__(self, args):
        self.dataset = Dataset(args)
        self.dataset_name = args.dataset
        self.experiment_name = args.experiment_name
        self.adv = args.adv
        self.model_name = args.model
        if self.adv:
            self.model = AMR(args, self.dataset.usz, self.dataset.isz, self.dataset.fsz)
            self.sess = tf.compat.v1.Session()
            self.sess.run(tf.compat.v1.global_variables_initializer())
            self.saver = tf.compat.v1.train.Saver(tf.compat.v1.trainable_variables(), max_to_keep=0)
            self.sess.run(self.model.assign_image, feed_dict={self.model.init_image: self.dataset.emb_image})
        else:
            if self.model_name == 'VBPR':
                self.model = VBPR(args, self.dataset.usz, self.dataset.isz, self.dataset.fsz)
                self.sess = tf.compat.v1.Session()
                self.sess.run(tf.compat.v1.global_variables_initializer())
                self.saver = tf.compat.v1.train.Saver(tf.compat.v1.trainable_variables(), max_to_keep=0)
                self.sess.run(self.model.assign_image, feed_dict={self.model.init_image: self.dataset.emb_image})
            elif self.model_name == 'DVBPR':
                self.model = DVBPR(args, self.dataset.usz, self.dataset.isz)
            else:
                raise NotImplemented('The model has not been implemented yet!')
        self.epoch = args.epoch
        self.verbose = args.verbose



        self.topk = args.topk
        self.weight_dir = '../' + args.weight_dir + '/'
        self.result_dir = '../' + args.result_dir + '/'

        if self.experiment_name in ['original', 'madry_original', 'free_adv_original']:
            self.defense_type = ''
            self.attack_type = 'original'
            self.attacked_categories = ''
            self.eps_cnn = ''
            self.iteration_attack_type = ''
            self.norm = ''
        else:
            self.defense_type = self.experiment_name.split('_')[0]
            self.attack_type = self.experiment_name.split('_')[1]
            self.attacked_categories = '_' + self.experiment_name.split('_')[2] + '_' + self.experiment_name.split('_')[
                3]
            self.eps_cnn = '_' + self.experiment_name.split('_')[4]
            self.iteration_attack_type = '_' + self.experiment_name.split('_')[5]
            self.norm = '_' + self.experiment_name.split('_')[6]

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

    def one_epoch_tf2(self):
        loss = 0.0
        steps = 0
        generator = self.dataset.batch_generator()
        while True:
            try:
                steps += 1
                train_inputs = next(generator)
                loss += self.model.train_step(*train_inputs)
            except StopIteration:
                break
        return loss / steps

    def train(self):
        start_epoch = 0
        if self.adv:
            start_epoch = self.epoch // 2

        for i in range(start_epoch + 1, self.epoch + 1):
            start = time.time()

            if self.model_name == 'DVBPR':
                self.one_epoch_tf2()
            else:
                self.one_epoch()
            if i % self.verbose == 0:
                self.save(i)
            print('Epoch {0}/{1} in {2} secs.'.format(i, self.epoch, time.time() - start))

        if self.model_name == 'VBPR' or self.adv:
            self.new_store_predictions(i)
        else:
            self.new_store_predictions_tf_2(i)

        if self.model_name == 'VBPR' or self.adv:
            self.save(i)
        else:
            self.save_tf_2(i)

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
        emb_P = self.sess.run(self.model.emb_P) * -1
        temp_emb_Q = self.sess.run(self.model.temp_emb_Q)
        predictions = np.matmul(emb_P, temp_emb_Q.transpose())
        # Store Prediction Positions
        position_predictions = predictions.argsort(axis=1)
        position_predictions = [position_predictions[i][:self.topk] for i in
                                range(position_predictions.shape[0])]
        prediction_name = self.result_dir + self.experiment_name + '_top{0}_pos_ep{1}'.format(
            self.topk, epoch)
        if self.adv:
            prediction_name = prediction_name + '_AMR'
        else:
            prediction_name = prediction_name + '_VBPR'

        write.save_obj(position_predictions, prediction_name)

        # Store Prediction Scores
        score_predictions = predictions.argsort(axis=1)
        score_predictions = [score_predictions[i][:self.topk] for i in range(score_predictions.shape[0])]
        prediction_name = self.result_dir + self.experiment_name + '_top{0}_score_ep{1}'.format(
            self.topk, epoch)
        if self.adv:
            prediction_name = prediction_name + '_AMR'
        else:
            prediction_name = prediction_name + '_VBPR'

        write.save_obj(score_predictions, prediction_name)

        print('End Store Predictions {0}'.format(time.time() - start))

    def new_store_predictions_tf_2(self, epoch):
        print('Start Store Predictions at epoch {0}'.format(epoch))
        start = time.time()

        predictions = self.model.prediction_all().numpy()

        if self.model_name == 'DVBPR':
            np.save(features_DVBPR_path.format(self.dataset_name), self.model.Phi)

        print("Storing results...")

        with open(self.result_dir + self.experiment_name + '_top{0}_ep{1}_{2}.tsv'.format(self.topk,
                                                                                          epoch,
                                                                                          self.model_name),
                  'w') as out:
            for user_id, u_predictions in enumerate(predictions):
                # for other metrics calculation
                u_predictions[self.dataset.df_train[self.dataset.df_train[0] == user_id][1].to_list()] = -np.inf
                top_k_id = u_predictions.argsort()[-self.topk:][::-1]
                top_k_score = u_predictions[top_k_id]
                for i, item_id in enumerate(top_k_id):
                    out.write(str(user_id) + '\t' + str(item_id) + '\t' + str(top_k_score[i]) + '\n')

        print('End Store Predictions in {0} seconds'.format(time.time() - start))

    def new_store_predictions(self, epoch):
        # We multiply the users embeddings by -1 to have the np sorting operation in the correct order

        print('Start Store Predictions at epoch {0}'.format(epoch))
        start = time.time()
        # predictions = self.sess.run(self.model.predictions)
        emb_P = self.sess.run(self.model.emb_P)
        temp_emb_Q = self.sess.run(self.model.temp_emb_Q)
        predictions = np.matmul(emb_P, temp_emb_Q.transpose())

        print("Storing results...")

        with open(self.result_dir + self.experiment_name + '_top{0}_ep{1}_{2}.tsv'.format(self.topk, epoch, 'AMR' if self.adv else 'VBPR'), 'w') as out:
            for user_id, u_predictions in enumerate(predictions):
                u_predictions[self.dataset.df_train[self.dataset.df_train[0] == user_id][1].to_list()] = -np.inf
                top_k_id = u_predictions.argsort()[-self.topk:][::-1]
                top_k_score = u_predictions[top_k_id]
                for i, item_id in enumerate(top_k_id):
                    out.write(str(user_id) + '\t' + str(item_id) + '\t' + str(top_k_score[i]) + '\n')

        print('End Store Predictions {0}'.format(time.time() - start))

    def load(self):
        try:
            params = np.load(self.weight_dir + self.experiment_name + '_step{0}_VBPR.npy'.format(self.epoch // 2),
                             allow_pickle=True)
            self.sess.run([self.model.assign_P, self.model.assign_Q, self.model.phi.assign(params[2])],
                          {self.model.init_emb_P: params[0], self.model.init_emb_Q: params[1]})
            print('Load parameters from {0}'.format(self.weight_dir + self.experiment_name + '_step{0}_VBPR.npy'.format(self.epoch // 2)))
        except Exception as ex:
            print('Start new model from scratch')

    def save_tf_2(self, epoch):
        params, optimizer = self.model.get_model_params()
        store_model_path = self.weight_dir + self.experiment_name + '_ep{0}_{1}'.format(epoch, self.model_name)

        # save model state as dict
        model_dict = {
            'model_name': self.model_name,
            'dataset': self.dataset_name,
            'epoch': epoch,
            'model_state_dict': params,
            'optimizer_state_np': optimizer.get_weights()
        }

        with open(store_model_path + '.pkl', 'wb') as f:
            pickle.dump(model_dict, f)

    def save(self, step):
        params = self.sess.run(tf.compat.v1.trainable_variables())
        store_model_path = self.weight_dir + self.experiment_name + '_step{0}'.format(step)
        if self.adv:
            store_model_path = store_model_path + '_AMR'
        else:
            store_model_path = store_model_path + '_VBPR'

        np.save(store_model_path + '.npy', params)

