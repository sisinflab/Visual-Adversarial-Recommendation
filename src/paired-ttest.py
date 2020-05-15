import argparse
import pandas as pd
import time
import multiprocessing as mp
import os
import math
from scipy import stats

counter = 0
start_counter = 0
users_size = 0
classes = pd.DataFrame()
import typing


def elaborate_chr(class_frequency, user_id, sorted_item_predictions, origin):
    """
    Methos to elaborate the prediction (CHR@K) for each user
    CHR@N(I_c, U) = \frac{1}{N \cdot |U|} \sum_{u \in U} \sum_{i \in I_c \setminus I^{+}_u} hit(i, u)
    :param class_frequency:
    :param user_id:
    :param sorted_item_predictions:
    :return: user id sent to the count-elaborated
    """
    # Count the class occurrences for the user: user_id
    for item_index in sorted_item_predictions:
        item_original_class = classes[classes['ImageID'] == item_index]['ClassNum'].values[0]
        if item_original_class == origin:
            class_frequency[user_id] += 1
    return user_id


def compute_ndcg(sorted_item_predictions: typing.List, gain_map: typing.Dict, cutoff: int) -> float:
    """
    Method to compute nDCG
    :param sorted_item_predictions:
    :param gain_map:
    :param cutoff:
    :return:
    """
    idcg: float = compute_idcg(gain_map, cutoff)
    dcg: float = sum(
        [gain_map.get(x, 0) * compute_dicount(r) for r, x in enumerate(sorted_item_predictions) if r < cutoff])
    return dcg / idcg if dcg > 0 else 0


def compute_idcg(gain_map: typing.Dict, cutoff: int) -> float:
    """
    Method to compute Ideal Discounted Cumulative Gain
    :param gain_map:
    :param cutoff:
    :return:
    """
    gains: typing.List = sorted(list(gain_map.values()))
    n: int = min(len(gains), cutoff)
    m: int = len(gains)
    return sum(map(lambda g, r: gains[m - r - 1] * compute_dicount(r), gains, range(n)))


def compute_user_gain_map(sorted_item_predictions: typing.List, sorted_item_scores: typing.List,
                          threshold: int = 0) -> typing.Dict:
    """
    Method to compute the Gain Map:
    rel = 2**(score - threshold + 1) - 1
    :param sorted_item_predictions:
    :param sorted_item_scores:
    :param threshold:
    :return:
    """
    return {id: 0 if score < threshold else 2 ** (score - threshold + 1) - 1 for id, score in
            zip(sorted_item_predictions, sorted_item_scores)}
    # return {id: 0 if score < threshold else 2**(3) - 1 for id, score in zip(sorted_item_predictions, sorted_item_scores)}


def compute_category_user_gain_map(category_items: typing.List, threshold: int = 0) -> typing.Dict:
    """
    Method that computes the user gain map considering a list of category items with score 1
    :param sorted_item_predictions:
    :param sorted_item_scores:
    :param category_items:
    :param threshold:
    :return:
    """
    return {id: 2 ** (1 - threshold + 1) - 1 for id in category_items}


def compute_dicount(k: int) -> float:
    """
    Method to compute logarithmic discount
    :param k:
    :return:
    """
    return 1 / math.log(k + 2) * math.log(2)


def elaborate_cndcg(class_frequency, user_id, sorted_item_predictions, sorted_item_scores, positive_items,
                    category_items, item_original_class):
    """
    Methos to elaborate the prediction (CnDCG@K) for each user
    CnDCG@N(I_c, U) = \frac{1}{|U|} \sum_{u \in U}
        \Bigg(
            \frac
            {\sum_{pos = 1}^{|I_c \setminus I^{+}_u|} \frac{rel(u, pos)}{\log(pos + 1)}}
            {\sum_{pos = 1}^{|I_c \setminus I^{+}_u|} \frac{1}{\log(pos + 1)}}
        \Bigg)
    :param class_frequency:
    :param user_id:
    :param sorted_item_predictions:
    :param positive_items:
    :return: user id sent to the count-elaborated
    """

    # nDCG computed on training set
    # gain_map: typing.Dict = compute_user_gain_map(sorted_item_predictions, sorted_item_scores, 0)
    # ndcg: float = compute_ndcg(sorted_item_predictions, gain_map, len(sorted_item_predictions))

    # nDCG computed on training set considering a relevance based on categories
    gain_map: typing.Dict = compute_category_user_gain_map(category_items, 0)
    ndcg: float = compute_ndcg(sorted_item_predictions, gain_map, len(sorted_item_predictions))

    class_frequency[user_id] += ndcg

    return user_id


def count_elaborated(r):
    """
    Method to keep track of evaluation
    :param r:
    :return:
    """
    global counter, start_counter, users_size
    counter += 1
    if (counter + 1) % 1000 == 0:
        print('{0}/{1} in {2}'.format(counter + 1, users_size, time.time() - start_counter))
        start_counter = time.time()


def parse_args():
    parser = argparse.ArgumentParser(description="Run Recommender Model.")
    parser.add_argument('--dataset', nargs='?', default='tradesy', help='amazon_men, amazon_women, amazon_sport')
    parser.add_argument('--metric', nargs='?', default='cndcg', help='chr, cndcg')
    parser.add_argument('--experiment_name', nargs='?', default='original', help='original, fgsm_***, cw_***, pgd_***')
    parser.add_argument('--topk', type=int, default=150, help='top k predictions to store before the evaluation')
    parser.add_argument('--origin', type=int, default=834, help='Target Item id. Useful for CnDCG')
    parser.add_argument('--analyzed_k', type=int, default=20, help='K under analysis has to be lesser than stored topk')
    parser.add_argument('--num_pool', type=int, default=1,
                        help='Number of threads')

    return parser.parse_args()


def get_classes_dir(prediction_file):
    """
    Get the name of experiment original
    :param prediction_file:
    :return: original, madry_original, free_adv_original
    """
    if 'madry' in prediction_file:
        return 'madry_original'
    elif 'free' in prediction_file:
        return 'free_adv_original'
    else:
        return 'original'


if __name__ == '__main__':

    args = parse_args()

    with open('ttest.txt', 'w') as f:
        # Global Configuration
        for dataset in ['tradesy', 'amazon_men', 'amazon_women']:

            origin = 0
            if dataset == 'amazon_men':
                origin = 774
            elif dataset == 'amazon_women':
                origin = 610
            elif dataset == 'tradesy':
                origin = 834

            for an_metric in ['chr', 'cndcg']:
                for analyzed_k in [20, 50]:

                    ttest_map = {}

                    result_dir = '../rec_results/'
                    metric_dir = '../{0}/'.format(an_metric)
                    dataset_name = dataset
                    prediction_files_path = result_dir + dataset_name
                    N = 50  # Top-N classes

                    assert analyzed_k < args.topk

                    prediction_files = os.listdir(prediction_files_path)

                    df_ordered = pd.DataFrame(columns=['experiment', 'classId', 'className', 'position', 'score'])

                    for prediction_file in prediction_files:
                        if 'spsa' not in prediction_file and 'zoo' not in prediction_file:

                            counter = 0
                            start_counter = time.time()
                            start = time.time()

                            print('Analyzing {0} of {1}'.format(prediction_file, dataset_name))

                            predictions = pd.read_csv('../rec_results/{0}/{1}'.format(dataset_name, prediction_file),
                                                      sep='\t',
                                                      header=None)

                            # train = pd.read_csv('../data/{0}/trainingset.tsv'.format(dataset_name), sep='\t', header=None)
                            classes = pd.read_csv(
                                '../data/{0}/{1}/classes.csv'.format(dataset_name, get_classes_dir(prediction_file)))
                            category_items = classes[classes['ClassNum'] == origin]['ImageID'].to_list()

                            users_size = predictions[0].nunique()
                            train = pd.read_csv('../data/{0}/trainingset.tsv'.format(dataset_name), sep='\t',
                                                header=None)
                            train.columns = ['userId', 'itemId']

                            manager = mp.Manager()
                            class_frequency = manager.dict()
                            for user_id in train['userId'].unique():
                                class_frequency[user_id] = 0

                            p = mp.Pool(args.num_pool)

                            # Evaluate everything with respect to the origin (under evaluation) class
                            if an_metric == 'chr':
                                for user_id in predictions[0].unique():
                                    p.apply_async(elaborate_chr,
                                                  args=(class_frequency, user_id,
                                                        predictions[predictions[0] == user_id][1].to_list()[
                                                        :analyzed_k],
                                                        origin,),
                                                  callback=count_elaborated)
                            else:
                                for user_id in predictions[0].unique():
                                    p.apply_async(elaborate_cndcg,
                                                  args=(class_frequency, user_id,
                                                        predictions[predictions[0] == user_id][1].to_list()[
                                                        :analyzed_k],
                                                        predictions[predictions[0] == user_id][2].to_list()[
                                                        :analyzed_k],
                                                        train[train['userId'] == user_id]['itemId'].to_list(),
                                                        category_items,
                                                        origin,),
                                                  callback=count_elaborated)

                            p.close()
                            p.join()

                            # We need this operation to use the results in the Manager
                            metric = dict()
                            for key in class_frequency.keys():
                                metric[key] = class_frequency[key]

                            ttest_map[prediction_file] = metric

                    for experiment_name in ttest_map.keys():

                        recommender = None
                        if 'VBPR' in prediction_file:
                            recommender = 'VBPR'
                        elif 'AMR' in prediction_file:
                            recommender = 'AMR'
                        elif 'LFM' in prediction_file:
                            recommender = 'LFM'

                        baseline = None
                        if 'madry' in prediction_file:
                            baseline = "madry_original_top150_ep{0}_{1}.tsv".format(
                                100 if recommender == 'LFM' else 4000,
                                recommender)
                        elif 'free' in prediction_file:
                            baseline = 'free_adv_original_top150_ep{0}_{1}.tsv'.format(
                                100 if recommender == 'LFM' else 4000,
                                recommender)
                        else:
                            baseline = 'original_top150_ep{0}_{1}.tsv'.format(100 if recommender == 'LFM' else 4000,
                                                                              recommender)

                        baseline_experiment = None
                        if experiment_name.startswith('madry_original') or experiment_name.startswith(
                                'free_adv_original') or experiment_name.startswith('original'):
                            print('this element is teh baseline and we do not evaluate with respect to itself')
                        elif 'spsa' in experiment_name or 'zoo' in experiment_name:
                            print('Spsa e Zoo not under evaluation')
                        else:
                            # Take the baseline and compare with the model
                            baseline = ttest_map[baseline]
                            actual_experiment = ttest_map[experiment_name]

                            base = []
                            test = []

                            for user_id in actual_experiment.keys():
                                base.append(baseline[user_id])
                                test.append(actual_experiment[user_id])

                            p = stats.ttest_rel(base, test).pvalue
                            star = ''
                            if p <= 0.05:
                                star = '*'
                            else:
                                start = '#'

                            line = '{0}\t{1}\t{2}\t{3}\t{4}\t{5}'.format(dataset_name, an_metric, analyzed_k,
                                                                         experiment_name,
                                                                         p, star)
                            f.writelines(line)
                            print(line)

