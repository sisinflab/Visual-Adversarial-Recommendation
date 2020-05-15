import utils.read as read
import utils.write as write
from utils import get_server_name, cpu_count
from operator import itemgetter
import argparse

import pandas as pd
import time
import multiprocessing as mp
import os

counter = 0
start_counter = 0
users_size = 0
K = 0


def elaborate(class_frequency, user_id, user_positive_items, sorted_item_predictions):
    # Count the class occurrences for the user: user_id
    global K
    k = 0
    for item_index in sorted_item_predictions:
        if item_index not in user_positive_items:

            item_original_class = item_classes[item_classes['ImageID'] == item_index]['ClassStr'].values[0]

            class_frequency[item_original_class] += 1
            k += 1
            if k == K:
                break
    if k < K:
        print('User: {0} has more than {1} positive rated items in his/her top K'.format(user_id, K))

    return user_id


def count_elaborated(r):
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

    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()

    # Global Configuration
    result_dir = '../rec_results/'
    dataset_name = args.dataset
    experiment_name = args.experiment_name
    prediction_files_path = result_dir + dataset_name

    assert args.analyzed_k >= args.topk

    K = args.analyzed_k

    counter = 0
    start_counter = time.time()
    start = time.time()

    prediction_files = os.listdir(prediction_files_path)

    with open('results_test.txt', 'w') as f:

        for prediction_file in prediction_files:
            if not prediction_file.startswith('Top') and not prediction_file.startswith(
                    'Plot') and 'pos' in prediction_file:
                print(prediction_file)
                predictions = read.load_obj(prediction_files_path + prediction_file)

                pos_elements = pd.read_csv('../data/{0}/trainingset.txt'.format(dataset_name), sep='\t', header=None)
                pos_elements.columns = ['u', 'i']
                pos_elements.u = pos_elements.u.astype(int)
                pos_elements.i = pos_elements.i.astype(int)
                pos_elements = pos_elements.sort_values(by=['u', 'i'])

                if prediction_file.startswith('original'):
                    original_dir = 'original'
                elif prediction_file.startswith('madry'):
                    original_dir = 'madry_original'
                elif prediction_file.startswith('free'):
                    original_dir = 'free_adv_original'
                else:
                    print('Unknown Source')
                    exit()

                # I have to read always from ORIGINAL because it is the attack against that original target class
                item_classes = pd.read_csv('../data/{0}/{1}/classes.csv'.format(dataset_name, original_dir))

                manager = mp.Manager()
                class_frequency = manager.dict()
                for item_class in item_classes['ClassStr'].unique():
                    class_frequency[item_class] = 0

                users_size = len(predictions)

                p = mp.Pool(cpu_count() - 1)

                for user_id, sorted_item_predictions in enumerate(predictions):
                    user_positive_items = pos_elements[pos_elements['u'] == user_id]['i'].to_list()
                    p.apply_async(elaborate,
                                  args=(class_frequency, user_id, user_positive_items, sorted_item_predictions,),
                                  callback=count_elaborated)

                p.close()
                p.join()

                print('END in {0} - {1}'.format(time.time() - start, max(class_frequency.values())))

                # We need this operation to use the results in the Manager
                novel = dict()
                for key in class_frequency.keys():
                    novel[key] = class_frequency[key]

                # print(novel.items())

                N_USERS = pos_elements['u'].nunique()
                N = 50  # Top-N classes
                class_str_length = 10

                # Store class frequencies results
                class_frequency_file_name = 'Top{0}/Top{0}_class_frequency_of_'.format(K) + prediction_file.split('.')[
                    0]
                write.save_obj(novel, prediction_files_path + class_frequency_file_name)

                res = dict(sorted(novel.items(), key=itemgetter(1), reverse=True)[:N])

                res = {str(k)[:class_str_length]: v / N_USERS for k, v in res.items()}

                keys = res.keys()
                values = res.values()

                ordered = pd.DataFrame(list(zip(keys, values)), columns=['x', 'y']).sort_values(by=['y'],
                                                                                                ascending=False)

                print('\nExperiment Name: {0}'.format(prediction_file))
                print(ordered)

                f.writelines('\nExperiment Name: {0}'.format(prediction_file))
                f.writelines(ordered.to_string())