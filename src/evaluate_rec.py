from utils.sendmail import sendmail
from operator import itemgetter
import argparse
import pandas as pd
import time
import multiprocessing as mp
import os

counter = 0
start_counter = 0
users_size = 0
classes = pd.DataFrame()


def elaborate(class_frequency, user_id, sorted_item_predictions):
    """
    Methos to elaborate the prediction (CHR@K) for each user
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
    parser.add_argument('--dataset', nargs='?', default='amazon_men', help='amazon_men, amazon_women, amazon_sport')
    parser.add_argument('--experiment_name', nargs='?', default='original', help='original, fgsm_***, cw_***, pgd_***')
    parser.add_argument('--topk', type=int, default=150, help='top k predictions to store before the evaluation')
    parser.add_argument('--analyzed_k', type=int, default=100,
                        help='K under analysis has to be lesser than stored topk')
    parser.add_argument('--num_pool', type=int, default=6,
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
    chr_dir = '../chr/'
    dataset_name = args.dataset
    experiment_name = args.experiment_name
    prediction_files_path = result_dir + dataset_name

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

        manager = mp.Manager()
        class_frequency = manager.dict()
        for item_class in classes['ClassNum'].unique():
            class_frequency[item_class] = 0

        p = mp.Pool(args.num_pool)

        users_size = predictions[0].nunique()

        for user_id in predictions[0].unique():
            p.apply_async(elaborate,
                          args=(class_frequency, user_id,
                                predictions[predictions[0] == user_id][1].to_list()[:args.analyzed_k],),
                          callback=count_elaborated)

        p.close()
        p.join()

        # We need this operation to use the results in the Manager
        chr = dict()
        for key in class_frequency.keys():
            chr[key] = class_frequency[key]

        print('\tEvaluate CHR@{0}'.format(args.analyzed_k))
        N_USERS = predictions[0].nunique()
        N = 30  # Top-N classes
        res = dict(sorted(chr.items(), key=itemgetter(1), reverse=True)[:N])

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
            row['position'] = index+1
            row['className'] = classes[classes['ClassNum'] == int(row['classId'])].iloc[0]['ClassStr']
            temp_ordered.loc[index] = row

        df_ordered = df_ordered.append(temp_ordered[['experiment', 'classId', 'className', 'position', 'score']], ignore_index=True)

    df_ordered.to_csv('{0}{1}/df_category_hit_ratio.csv'.format(chr_dir, dataset_name), index=None)

    sendmail('Finish Amazon Men Baseline Evaluation', 'Finished!')
