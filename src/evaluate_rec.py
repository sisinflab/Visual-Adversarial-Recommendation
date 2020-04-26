from utils.sendmail import sendmail
from operator import itemgetter
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


def elaborate_chr(class_frequency, user_id, sorted_item_predictions):
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
        class_frequency[item_original_class] += 1

    return user_id


def elaborate_cndcg(class_frequency, user_id, sorted_item_predictions, positive_items, category_items, item_original_class):
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
    k = len(sorted_item_predictions)
    ik = len(set(category_items).difference(positive_items))
    upper_bound_cidcg = k if k < ik else ik

    category_iDCG = sum([1/math.log2(pos + 1) for pos in range(1, upper_bound_cidcg+1)])

    temp_category_DCG = []
    # Count the class occurrences for the user: user_id
    for pos, item_index in enumerate(sorted_item_predictions):
        if item_index in category_items:
            temp_category_DCG.append(1/math.log2(pos + 1 + 1)) # we need +1 +1 since the posiiton start from 0 in enumerate

    category_DCG = sum(temp_category_DCG)

    class_frequency[item_original_class] += category_DCG/category_iDCG

    return user_id


def count_elaborated(r):
    """
    Method to keep track of evaluation
    :param r:
    :return:
    """
    global counter, start_counter, users_size
    counter += 1
    if (counter + 1) % 100 == 0:
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

    # Global Configuration

    result_dir = '../rec_results/'
    metric_dir = '../{0}/'.format(args.metric)
    dataset_name = args.dataset
    experiment_name = args.experiment_name
    prediction_files_path = result_dir + dataset_name
    N = 50  # Top-N classes

    assert args.analyzed_k < args.topk

    prediction_files = os.listdir(prediction_files_path)

    df_ordered = pd.DataFrame(columns=['experiment', 'classId', 'className', 'position', 'score'])

    for prediction_file in prediction_files:

        counter = 0
        start_counter = time.time()
        start = time.time()

        print('Analyzing {0} of {1}'.format(prediction_file, dataset_name))

        predictions = pd.read_csv('../rec_results/{0}/{1}'.format(dataset_name, prediction_file), sep='\t',
                                  header=None)

        # train = pd.read_csv('../data/{0}/trainingset.tsv'.format(dataset_name), sep='\t', header=None)
        classes = pd.read_csv(
            '../data/{0}/{1}/classes.csv'.format(dataset_name, get_classes_dir(prediction_file)))
        users_size = predictions[0].nunique()

        if args.metric == 'chr':

            manager = mp.Manager()
            class_frequency = manager.dict()
            for item_class in classes['ClassNum'].unique():
                class_frequency[item_class] = 0

            p = mp.Pool(args.num_pool)

            for user_id in predictions[0].unique():
                p.apply_async(elaborate_chr,
                              args=(class_frequency, user_id,
                                    predictions[predictions[0] == user_id][1].to_list()[:args.analyzed_k],),
                              callback=count_elaborated)
        else:

            train = pd.read_csv('../data/{0}/trainingset.tsv'.format(dataset_name), sep='\t', header=None)
            train.columns = ['userId', 'itemId']

            category_items = classes[classes['ClassNum'] == args.origin]['ImageID'].to_list()

            manager = mp.Manager()
            class_frequency = manager.dict()
            class_frequency[args.origin] = 0

            p = mp.Pool(args.num_pool)

            for user_id in predictions[0].unique()[:50]:
                p.apply_async(elaborate_cndcg,
                              args=(class_frequency, user_id,
                                    predictions[predictions[0] == user_id][1].to_list()[:args.analyzed_k],
                                    train[train['userId'] == user_id]['itemId'].to_list(), category_items, args.origin, ),
                              callback=count_elaborated)

        p.close()
        p.join()

        # We need this operation to use the results in the Manager
        metric = dict()
        for key in class_frequency.keys():
            metric[key] = class_frequency[key]

        print('\tEvaluate {0}@{1}'.format(args.metric, args.analyzed_k))
        N_USERS = predictions[0].nunique()
        res = dict(sorted(metric.items(), key=itemgetter(1), reverse=True)[:N])

        res = {str(k): v / N_USERS for k, v in res.items()}

        keys = res.keys()
        values = res.values()

        temp_ordered = pd.DataFrame(list(zip(keys, values)), columns=['classId', 'score']).sort_values(by=['score'],
                                                                                                       ascending=False)
        print('\nExperiment Name: {0}'.format(prediction_file))

        temp_ordered['experiment'] = prediction_file
        temp_ordered['className'] = 0
        temp_ordered['position'] = 0

        for index, row in temp_ordered.iterrows():
            row['position'] = index + 1
            row['className'] = classes[classes['ClassNum'] == int(row['classId'])].iloc[0]['ClassStr']
            temp_ordered.loc[index] = row

        df_ordered = df_ordered.append(temp_ordered[['experiment', 'classId', 'className', 'position', 'score']],
                                       ignore_index=True)

    df_ordered.to_csv('{0}{1}/df_{2}_at_{3}.csv'.format(metric_dir, dataset_name, args.metric, args.analyzed_k),
                      index=None)

    sendmail('Finish {0} at Evaluation {1}@{2}'.format(dataset_name, args.metric, args.analyzed_k), 'Finished!')
