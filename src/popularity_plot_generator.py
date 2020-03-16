import os
import matplotlib.pyplot as plt
from operator import itemgetter
import pandas as pd
import utils.read as read

# Global Configuration
result_dir = '../rec_results/'
dataset_name = 'amazon_men/'
experiment_name = ''
prediction_files_path = result_dir + dataset_name

pos_elements = pd.read_csv('../data/{0}/pos.txt'.format(dataset_name), sep='\t', header=None)
pos_elements.columns = ['u', 'i']

N_USERS = pos_elements['u'].nunique()
K = 100
N = 30  # Top-N classes
class_str_length = 10

if __name__ == '__main__':

    prediction_files = os.listdir(prediction_files_path)

    for prediction_file in prediction_files:

        if prediction_file.startswith('Top'):
            class_frequencies = read.load_obj(prediction_files_path + prediction_file)

            res = dict(sorted(class_frequencies.items(), key=itemgetter(1), reverse=True)[:N])

            res = {str(k)[:class_str_length]: v / N_USERS for k, v in res.items()}

            keys = res.keys()
            values = res.values()

            ordered = pd.DataFrame(list(zip(keys, values)), columns=['x', 'y']).sort_values(by=['y'], ascending=False)

            print(ordered)

            plt.title('Top-{0} popular categories in top-{1}.'.format(N, K))

            # ordered.plot.bar(x='x', rot=70, label="Category distribution", align="center", color=color_map(data_normalizer(likeability_scores)))
            plt.bar(ordered['x'], ordered['y'], label="Category distribution", align="center")
            plt.xticks(rotation='70')

            plt.ylabel('avg # of items in top-{0}'.format(K))
            plt.xlabel('Item Category')

            plt.legend(bbox_to_anchor=(1, 1), loc="upper right", borderaxespad=0.5)

            plt.savefig(prediction_files_path + 'Plot_' + prediction_file.split('.')[0] + '.svg')
            plt.show()
