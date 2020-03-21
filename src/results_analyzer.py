import utils.read as read
import utils.write as write
from utils.sendmail import sendmail
from utils import get_server_name, cpu_count
from operator import itemgetter

import pandas as pd
import time
import multiprocessing as mp
import os


# Global Configuration
result_dir = '../rec_results/'
dataset_name = 'amazon_women/'
experiment_name = ''
tp_k_predictions = 1000
prediction_files_path = result_dir + dataset_name

K = 100
counter = 0
start_counter = time.time()
start = time.time()


def elaborate(class_frequency, user_id, user_positive_items, sorted_item_predictions):
    # Count the class occurrences for the user: user_id
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


if __name__ == '__main__':

    prediction_files = os.listdir(prediction_files_path)

    with open('results_aw_amr', 'w') as f:

        for prediction_file in prediction_files:
            # if not prediction_file.startswith('Top') and not prediction_file.startswith('Plot'):
            if not prediction_file.startswith('Top') and not prediction_file.startswith('Plot'):
                print(prediction_file)
                predictions = read.load_obj(prediction_files_path + prediction_file)

                pos_elements = pd.read_csv('../data/{0}/pos.txt'.format(dataset_name), sep='\t', header=None)
                pos_elements.columns = ['u', 'i']
                pos_elements.u = pos_elements.u.astype(int)
                pos_elements.i = pos_elements.i.astype(int)
                pos_elements = pos_elements.sort_values(by=['u', 'i'])

                item_classes = pd.read_csv('../data/{0}/original_images/classes.csv'.format(dataset_name))

                manager = mp.Manager()
                class_frequency = manager.dict()
                for item_class in item_classes['ClassStr'].unique():
                    class_frequency[item_class] = 0

                users_size = len(predictions)

                p = mp.Pool(cpu_count()-1)

                for user_id, sorted_item_predictions in enumerate(predictions):
                    user_positive_items = pos_elements[pos_elements['u'] == user_id]['i'].to_list()
                    p.apply_async(elaborate, args=(class_frequency, user_id, user_positive_items, sorted_item_predictions,),
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
                class_frequency_file_name = 'Top{0}/Top{0}_class_frequency_of_'.format(K) + prediction_file.split('.')[0]
                write.save_obj(novel, prediction_files_path + class_frequency_file_name)

                res = dict(sorted(novel.items(), key=itemgetter(1), reverse=True)[:N])

                res = {str(k)[:class_str_length]: v / N_USERS for k, v in res.items()}

                keys = res.keys()
                values = res.values()

                ordered = pd.DataFrame(list(zip(keys, values)), columns=['x', 'y']).sort_values(by=['y'], ascending=False)

                print('\nExperiment Name: {0}'.format(prediction_file))
                print(ordered)

                f.writelines('\nExperiment Name: {0}'.format(prediction_file))
                f.writelines(ordered.to_string())

    sendmail('Elaborate Predictions on {0}'.format(get_server_name()), 'Amazon Women')

