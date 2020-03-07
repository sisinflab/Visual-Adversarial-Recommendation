import utils.read as read
import utils.write as write
from utils.sendmail import sendmail
from utils import get_server_name

import pandas as pd
import time
import multiprocessing as mp
import os


# Global Configuration
result_dir = '../rec_results/'
dataset_name = 'amazon_men/'
experiment_name = ''
tp_k_predictions = 300
prediction_files_path = result_dir + dataset_name

K = 100
counter = 0
start_counter = time.time()
start = time.time()


def elaborate(class_frequency, user_id, user_positive_items, sorted_item_predictions):
    start_users = time.time()
    # Count the class occurrences
    k = 0
    for item_index in sorted_item_predictions:
        if item_index not in user_positive_items:
            item_original_id = item_indices[item_index]
            item_original_class = item_classes[item_classes['ImageID'] == item_original_id]['Class'].values[0]
            class_frequency[item_original_class] += 1
            k += 1

            if k == K:
                # ENd Top-K
                break
    if k < K:
        print('User: {0} has more than {1} positive rated items in his/her top K'.format(user_id, K))

    print('\t{0} in {1}'.format(user_id, time.time() - start_users))

    return user_id


def count_elaborated(r):
    global counter, start_counter, users_size
    counter += 1
    if (counter + 1) % 100 == 0:
        print('{0}/{1} in {2}'.format(counter + 1, users_size, time.time() - start_counter))
        start_counter = time.time()


if __name__ == '__main__':

    prediction_files = os.listdir(prediction_files_path)

    for prediction_file in prediction_files:

        predictions = read.load_obj(prediction_files_path + prediction_file)

        pos_elements = pd.read_csv('../data/{0}/train.txt'.format(dataset_name), sep='\t', header=None)
        pos_elements.columns = ['u', 'i']
        pos_elements.u = pos_elements.u.astype(int)
        pos_elements.i = pos_elements.i.astype(int)

        pos_elements = pos_elements.sort_values(by=['u', 'i'])

        old_item_indices = read.load_obj('../data/{0}/item_indices'.format(dataset_name))

        item_indices = {}
        # cast to KEY: integer and VALUE: original index
        for original_id in old_item_indices.keys():
            item_indices[old_item_indices[original_id]] = str(original_id)

        item_classes = pd.read_csv('../data/{0}/men_classes.csv'.format(dataset_name))

        item_occurences = pd.read_csv('../data/{0}/men_occurences.csv'.format(dataset_name))

        manager = mp.Manager()
        class_frequency = manager.dict()
        for item_class in item_occurences['Class'].unique():
            class_frequency[item_class] = 0

        users_size = len(predictions)

        p = mp.Pool(11)

        for user_id, sorted_item_predictions in enumerate(predictions):
            # TODO Remove already rated from the top-k

            user_positive_items = pos_elements[pos_elements['u'] == user_id]['i'].to_list()
            p.apply_async(elaborate, args=(class_frequency, user_id, user_positive_items, sorted_item_predictions,),
                          callback=count_elaborated)

        p.close()
        p.join()

        print('END in {0} - {1}'.format(time.time() - start, max(class_frequency.values())))

        novel = dict()
        for key in class_frequency.keys():
            novel[key] = class_frequency[key]

        write.save_obj(novel, 'fgsm_amazon_men_4000_epochs/CLASS_FREQ_epoch_4000_k_100')

        sendmail('Elaborate Predictions on {0}', 'Execution time: {1}'.format(get_server_name(), time.time() - start))

        print(novel.values())
