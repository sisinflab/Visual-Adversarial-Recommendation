import multiprocessing as mp
import time

import pandas as pd

import read
import write
import utils

counter = 0
start_counter = time.time()


def process_user_recommendations(class_frequency, user_id, user_positive_items, sorted_item_predictions):
    """
    Count the number of occurrences of each category in each user top-k recommendation list
    :param class_frequency: the global object where occ. are stored
    :param user_id: the current user
    :param user_positive_items: the user's rated item that will not be considered
    :param sorted_item_predictions: top recommended items (this list includes also the rated-items)
    :return: the user-id for who the elaboration of recommendations has been completed
    """
    k = 0
    for item_index in sorted_item_predictions:
        if item_index not in user_positive_items:
            item_original_id = item_indices[item_index]
            item_original_class = item_classes[item_classes['ImageID'] == item_original_id]['Class'].values[0]
            class_frequency[item_original_class] += 1
            k += 1

            if k == K:
                # End the analysis of the top-k items
                break
    if k < K:
        print('User: {0} has more than {1} positive rated items in his/her top K'.format(user_id, K))

    return user_id


def track_processed_users(r):
    global counter, start_counter, users_size
    counter += 1
    if (counter + 1) % 100 == 0:
        print('{0}/{1} in {2}'.format(counter + 1, users_size, time.time() - start_counter))
        start_counter = time.time()


if __name__ == '__main__':

    start = time.time()
    print('Start Elaborate Prediction')
    K = 100

    dataset = 'amazon_men'

    predictions = read.load_obj('fgsm_amazon_men_4000_epochs/firts-200-predictions-epoch_4000')

    pos_elements = pd.read_csv('../data/{0}/pos.txt'.format(dataset), sep='\t', header=None)
    pos_elements.columns = ['u', 'i']
    pos_elements.u = pos_elements.u.astype(int)
    pos_elements.i = pos_elements.i.astype(int)

    pos_elements = pos_elements.sort_values(by=['u', 'i'])

    old_item_indices = read.load_obj('../data/{0}/item_indices'.format(dataset))

    item_indices = {}
    # cast to KEY: integer and VALUE: original item ID
    for original_id in old_item_indices.keys():
        item_indices[old_item_indices[original_id]] = str(original_id)

    item_classes = pd.read_csv('../data/{0}/men_classes.csv'.format(dataset))

    item_occurences = pd.read_csv('../data/{0}/men_occurences.csv'.format(dataset))

    manager = mp.Manager()
    class_frequency = manager.dict()
    for item_class in item_occurences['Class'].unique():
        class_frequency[item_class] = 0

    users_size = len(predictions)

    p = mp.Pool(processes=utils.cpu_count())

    for user_id, sorted_item_predictions in enumerate(predictions):
        user_positive_items = pos_elements[pos_elements['u'] == user_id]['i'].to_list()
        p.apply_async(process_user_recommendations, args=(class_frequency, user_id, user_positive_items, sorted_item_predictions,),
                      callback=track_processed_users)

    p.close()
    p.join()

    print('END in {0} - {1}'.format(time.time() - start, max(class_frequency.values())))

    novel = dict()
    for key in class_frequency.keys():
        novel[key] = class_frequency[key]

    write.save_obj(novel, 'fgsm_amazon_men_4000_epochs/CLASS_FREQ_epoch_4000_k_100')
