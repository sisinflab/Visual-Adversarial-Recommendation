from operator import itemgetter
from scipy import stats
import argparse
import pandas as pd
import time
import multiprocessing as mp
import os
import math

counter = 0
start_counter = 0
users_size = 0
classes = pd.DataFrame()
import typing


def elaborate_chr(class_frequency, metric_per_user, origin, user_id, sorted_item_predictions):
    """
    Methos to elaborate the prediction (CHR@K) for each user
    CHR@N(I_c, U) = \frac{1}{N \cdot |U|} \sum_{u \in U} \sum_{i \in I_c \setminus I^{+}_u} hit(i, u)
    :param class_frequency:
    :param user_id:
    :param sorted_item_predictions:
    :return: user id sent to the count-elaborated
    """
    metric_per_user[user_id] = 0
    # Count the class occurrences for the user: user_id
    for item_index in sorted_item_predictions:
        item_original_class = classes[classes['ImageID'] == item_index]['ClassNum'].values[0]
        class_frequency[item_original_class] += 1
        if item_original_class == origin:
            metric_per_user[user_id] = 1

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


def elaborate_ncdcg(class_frequency, metric_per_user, origin, user_id, sorted_item_predictions, sorted_item_scores, positive_items,
                    category_items, item_original_class):
    """
    Method to elaborate the prediction (ncdcg@K) for each user

    :param class_frequency:
    :param user_id:
    :param sorted_item_predictions:
    :param positive_items:
    :return: user id sent to the count-elaborated
    """

    # nDCG computed on training set
    # gain_map: typing.Dict = compute_user_gain_map(sorted_item_predictions, sorted_item_scores, 0)
    # ndcg: float = compute_ndcg(sorted_item_predictions, gain_map, len(sorted_item_predictions))

    ## nDCG computed on training set considering a relevance based on categories
    metric_per_user[user_id] = 0
    gain_map: typing.Dict = compute_category_user_gain_map(category_items, 0)
    ndcg: float = compute_ndcg(sorted_item_predictions, gain_map, len(sorted_item_predictions))

    class_frequency[item_original_class] += ndcg
    if item_original_class == origin:
        metric_per_user[user_id] = ndcg

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
    parser.add_argument('--dataset', nargs='?', default='tradesy_original', help='amazon_men, amazon_women, amazon_sport')
    parser.add_argument('--metric', nargs='?', default='chr', help='chr, cndcg')
    parser.add_argument('--model', nargs='?', default='ACF', help='ACF, DVBPR')
    # parser.add_argument('--experiment_name', nargs='?', default='original', help='original, fgsm_***, cw_***, pgd_***')
    parser.add_argument('--topk', type=int, default=100, help='top k predictions to store before the evaluation')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs')
    parser.add_argument('--origin', type=int, default=834, help='Target Item id. Useful for cndcg')
    parser.add_argument('--num_pool', type=int, default=1,
                        help='Number of threads')
    parser.add_argument('--list_of_k', nargs='+', type=int, default=[1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                        help='list of top-k to evaluate on')

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

    # Global Configuration

    result_dir = '../rec_results/'
    metric_dir = '../{0}/'.format(args.metric)
    dataset_name = args.dataset
    # experiment_name = args.experiment_name
    prediction_files_path = result_dir + dataset_name
    N = 50  # Top-N classes

    prediction_files = os.listdir(prediction_files_path)
    prediction_files = [f for f in prediction_files if args.model in f and 'spsa' not in f and 'zoo' not in f and 'eps8' not in f]

    for current_top_k in list(args.list_of_k):
        print('***************************************************')
        print('ANALYZING STATISTICS FOR TOP-{0}'.format(current_top_k))
        print('***************************************************')

        df_ordered = pd.DataFrame([], columns=['experiment', 'classId', 'className', 'position', 'score', 'p-value'])
        # per ogni row calcoliamo il p-value
        # se row contiene original (anche difese), allora è baseline e non prendiamo p-value
        # se non è baseline, allora si trova baseline associata

        ttest_map = {}
        for prediction_file in prediction_files:

            counter = 0
            start_counter = time.time()
            start = time.time()

            print('Analyzing {0} of {1}'.format(prediction_file, dataset_name))

            predictions = pd.read_csv('../rec_results/{0}/{1}'.format(dataset_name, prediction_file), sep='\t',
                                      header=None)

            classes = pd.read_csv(
                '../data/{0}/{1}/classes.csv'.format(dataset_name, get_classes_dir(prediction_file)))
            users_size = predictions[0].nunique()

            if args.metric == 'chr':

                manager = mp.Manager()
                class_frequency = manager.dict()
                metric_per_user = manager.dict()
                for item_class in classes['ClassNum'].unique():
                    class_frequency[item_class] = 0

                p = mp.Pool(args.num_pool)

                for user_id in predictions[0].unique():
                    p.apply_async(elaborate_chr,
                                  args=(class_frequency, metric_per_user, args.origin, user_id,
                                        predictions[predictions[0] == user_id][1].to_list()[:current_top_k],),
                                  callback=count_elaborated)
            else:

                train = pd.read_csv('../data/{0}/trainingset.tsv'.format(dataset_name), sep='\t', header=None)
                train.columns = ['userId', 'itemId']

                category_items = classes[classes['ClassNum'] == args.origin]['ImageID'].to_list()

                manager = mp.Manager()
                metric_per_user = manager.dict()
                class_frequency = manager.dict()
                class_frequency[args.origin] = 0

                p = mp.Pool(args.num_pool)

                for user_id in predictions[0].unique():
                    p.apply_async(elaborate_ncdcg,
                                  args=(class_frequency, metric_per_user, args.origin, user_id,
                                        predictions[predictions[0] == user_id][1].to_list()[:current_top_k],
                                        predictions[predictions[0] == user_id][2].to_list()[:current_top_k],
                                        train[train['userId'] == user_id]['itemId'].to_list(), category_items,
                                        args.origin,),
                                  callback=count_elaborated)

            p.close()
            p.join()

            # We need this operation to use the results in the Manager
            metric = dict()
            metric_per_user_final = dict()

            for key in metric_per_user.keys():
                metric_per_user_final[key] = metric_per_user[key]

            for key in class_frequency.keys():
                print('Val {0}'.format(class_frequency[key]))
                metric[key] = class_frequency[key]

            ttest_map[prediction_file] = metric_per_user_final

            print('\tEvaluate {0}@{1}'.format(args.metric, current_top_k))
            N_USERS = predictions[0].nunique()
            res = dict(sorted(metric.items(), key=itemgetter(1), reverse=True)[:N])

            res = {str(k): v / N_USERS for k, v in res.items()}
            print(res)
            keys = res.keys()
            values = res.values()

            temp_ordered = pd.DataFrame(list(zip(keys, values)), columns=['classId', 'score']).sort_values(by=['score'],
                                                                                                           ascending=False)
            print('\nExperiment Name: {0}'.format(prediction_file))

            temp_ordered['experiment'] = prediction_file
            temp_ordered['className'] = 0
            temp_ordered['position'] = 0
            temp_ordered['p-value'] = ''

            for index, row in temp_ordered.iterrows():
                row['position'] = index + 1
                row['className'] = classes[classes['ClassNum'] == int(row['classId'])].iloc[0]['ClassStr']
                temp_ordered.loc[index] = row

            df_ordered = df_ordered.append(temp_ordered[
                                               ['experiment', 'classId', 'className', 'position', 'score', 'p-value']
                                           ],
                                           ignore_index=True)

        df_ordered.to_csv('{0}{1}/df_{2}_at_{3}_{4}.csv'.format(metric_dir,
                                                                dataset_name,
                                                                args.metric,
                                                                current_top_k,
                                                                args.model),
                          index=False)

        # When the metric has been calculated for all rows, get the p-value
        # Take the baseline and compare with the model
        experiments_no_baselines = [f for f in prediction_files if 'original' not in f]
        for enb in experiments_no_baselines:
            if 'madry' not in enb and 'free_adv' not in enb:
                correspondent_baseline = 'original_top' + str(args.topk) + '_ep' + str(args.epochs) + '_' + str(args.model) + '.tsv'
            else:
                correspondent_baseline = ('madry_' if 'madry' in enb else 'free_adv_') + \
                                         'original_top' + str(args.topk) + '_ep' + \
                                         str(args.epochs) + '_' + str(args.model) + '.tsv'
            baseline = ttest_map[correspondent_baseline]
            actual_experiment = ttest_map[enb]

            base = []
            test = []

            for user_id in actual_experiment.keys():
                base.append(baseline[user_id])
                test.append(actual_experiment[user_id])

            p = stats.ttest_rel(base, test).pvalue
            if p <= 0.05:
                index = df_ordered.index[df_ordered['experiment'] == enb]
                df_ordered.loc[index, 'p-value'] = '*'

            df_ordered.to_csv('{0}{1}/df_{2}_at_{3}_{4}.csv'.format(metric_dir,
                                                                    dataset_name,
                                                                    args.metric,
                                                                    current_top_k,
                                                                    args.model),
                              index=False)
        df_ordered.to_csv('{0}{1}/df_{2}_at_{3}_{4}.csv'.format(metric_dir,
                                                                dataset_name,
                                                                args.metric,
                                                                current_top_k,
                                                                args.model),
                          index=False)

        # sendmail('Finish {0} at Evaluation {1}@{2}'.format(dataset_name, args.metric, args.analyzed_k), 'Finished!')

