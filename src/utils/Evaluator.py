import numpy as np
from multiprocessing import Pool
from multiprocessing import cpu_count
import sys
import math

_feed_dict = None
_dataset = None
_model = None
_sess = None
_K = None


def _init_eval_model(data):
    global _dataset
    _dataset = data

    pool = Pool(cpu_count() - 1)
    feed_dicts = pool.map(_evaluate_input, range(_dataset.num_users))
    pool.close()
    pool.join()

    return feed_dicts


def _evaluate_input(user):
    # generate items_list
    test_item = _dataset.test[user][1]
    item_input = set(range(_dataset.num_items)) - set(_dataset.train_list[user])
    if test_item in item_input:
        item_input.remove(test_item)
    item_input = list(item_input)
    item_input.append(test_item)
    user_input = np.full(len(item_input), user, dtype='int32')[:, None]
    item_input = np.array(item_input)[:, None]
    return user_input, item_input


def _eval_by_user(user):
    # get predictions of data in testing set
    user_input, item_input = _feed_dicts[user]
    predictions, _, _ = _model.get_inference(user_input, item_input)

    neg_predict, pos_predict = predictions[:-1], predictions[-1]
    position = (neg_predict.numpy() >= pos_predict.numpy()).sum()

    # calculate from HR@1 to HR@100, and from NDCG@1 to NDCG@100, AUC
    hr, ndcg, auc = [], [], []
    for k in range(1, _K + 1):
        hr.append(position < k)
        ndcg.append(math.log(2) / math.log(position + 2) if position < k else 0)
        auc.append(
            1 - (position / len(neg_predict)))  # formula: [#(Xui>Xuj) / #(Items)] = [1 - #(Xui<=Xuj) / #(Items)]

    return hr, ndcg, auc


class Evaluator:
    def __init__(self, k):
        """
        Class to manage all the evaluation methods and operation
        :param data: dataset object
        :param k: top-k evaluation
        """
        self.k = k

    def eval(self, epoch=0, results={}, epoch_text=''):
        """
        Runtime Evaluation of Accuracy Performance (top-k)
        :return:
        """
        global _model
        global _K
        global _dataset
        global _feed_dicts
        _dataset = self.data
        _model = self.model
        _K = self.k
        _feed_dicts = self.eval_feed_dicts

        res = []
        for user in range(self.model.data.num_users):
            res.append(_eval_by_user(user))

        hr, ndcg, auc = (np.array(res).mean(axis=0)).tolist()
        print("%sPerformance@%d \tHR: %.4f\tnDCG: %.4f\tAUC: %.4f" % (
            epoch_text, _K, hr[_K - 1], ndcg[_K - 1], auc[_K - 1]))

        if len(epoch_text) != '':
            results[epoch] = {'hr': hr, 'ndcg': ndcg, 'auc': auc[0]}

    def store_recommendation(self, predictions, attack_name=""):
        """
        Store recommendation list (top-k) in order to be used for the ranksys framework (sisinflab)
        attack_name: The name for the attack stored file
        :return:
        """
        results = predictions
        with open('{0}/{1}-top{2}-rec.tsv'.format(self.model.path_output_rec_result,
                                                  attack_name + self.model.path_output_rec_result.split('/')[-2],
                                                  self.k),
                  'w') as out:
            for u in range(results.shape[0]):
                results[u][self.data.train_list[u]] = -np.inf
                top_k = results[u].argsort()[-self.k:][::-1]
                top_k_score = results[u][top_k]
                for i in range(len(top_k)):
                    out.write(str(u) + '\t' + str(i) + '\t' + str(top_k_score[i]) + '\n')

