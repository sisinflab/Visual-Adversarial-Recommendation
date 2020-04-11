from builtins import enumerate

import pandas as pd
from lightfm.data import Dataset

import utils.write as write
from utils.read import read_np
from lightfm import LightFM
from lightfm.evaluation import recall_at_k
from time import time
import numpy as np
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Run Recommender Model.")
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--dataset', nargs='?', default='amazon_women', help='dataset path')
    parser.add_argument('--experiment_name', nargs='?', default='original', help='original, fgsm_***, cw_***, pgd_***')
    parser.add_argument('--emb1_K', type=int, default=64, help='size of embeddings')
    parser.add_argument('--loss', nargs='?', default='bpr', help='loss of FM model: logistic, bpr, warp, warp-kos')
    parser.add_argument('--batch_size', type=int, default=512, help='batch size')
    parser.add_argument('--lr', type=float, default=0.05, help='learning rate')
    parser.add_argument('--verbose', type=int, default=1000, help='vebrosity and Checkpoint epoch')
    parser.add_argument('--epoch', type=int, default=100, help='epochs')
    parser.add_argument('--cnn', nargs='?', default='resnet', help='cnn type: resnet50')
    parser.add_argument('--weight_dir', nargs='?', default='rec_model_weights', help='directory to store the weights')
    parser.add_argument('--result_dir', nargs='?', default='rec_results', help='directory to store the predictions')
    parser.add_argument('--topk', type=int, default=150,
                        help='top k predictions to store before the evaluation')
    parser.add_argument('--num_threads', type=int, default=4, help='Number of parallel computation threads to use')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    print("raining of Light FM on DATASET {0}".format(args.dataset))
    print("- PARAMETERS:")
    for arg in vars(args):
        print("\t- " + str(arg) + " = " + str(getattr(args, arg)))
    print("\n")

    topk = args.topk
    dataset_directory = '../data/' + args.dataset
    result_directory = '../rec_results/{0}/{1}/'.format(args.dataset, args.experiment_name)

    # Read Dataset
    df_train = pd.read_csv(dataset_directory+'/trainingset.tsv', header=None, sep='\t')
    df_test = pd.read_csv(dataset_directory+'testset.tsv', header=None, sep='\t')
    features = read_np('{0}/{1}/features.npy'.format(dataset_directory, args.experiment_name))

    train = Dataset()
    train.fit(df_train[0].unique(), df_train[1].unique(), item_features=range(2048))
    # Build Train Interactions
    (train_interactions, weights) = train.build_interactions(((row[0], row[1])
                                                              for index, row in df_train.iterrows()))

    # Build Features
    # Call build_user/item_features with iterables of (user/item id, [features]) or (user/item id, {feature: feature weight}) to build feature matrices.
    print('Loading Features...')
    list_features = {}
    for image_index, image_feature in enumerate(features):
        list_features[image_index] = []
        for feature_index, feature_weight in enumerate(image_feature):
            list_features[image_index].append({feature_index: feature_weight})

    features_generator = ((item_id, ele) for item_id in list_features.keys() for ele in list_features[item_id])
    item_features = train.build_item_features(features_generator, normalize=False)
    print('End Loading Features.')

    # Train the Model
    print('Training...')
    start = time()
    model = LightFM(no_components=args.emb1_K, loss=args.loss, learning_rate=args.lr, random_state=0)
    model.fit(train_interactions, item_features=item_features, epochs=args.epochs, num_threads=args.num_threads, verbose=True)
    print('End Training in {0}.'.format(time() - start))

    # # Evaluation
    print("Start")
    full_users = []
    full_items = []
    for user_id in range(df_train[0].nunique()):
        full_users += [user_id] * df_train[1].nunique()
        full_items += range(df_train[1].nunique())
    full_users = np.array(full_users, dtype=np.int32)
    full_predictions = model.predict(full_users, full_items, item_features=item_features, num_threads=args.num_threads)

    # Store Prediction Lists (.tsv)
    with open(result_directory + '_top{0}_ep{1}_LFM.tsv'.format(args.topk, args.epoch), 'w') as out:
        num_items = df_train[1].nunique()
        num_users = df_train[0].nunique()
        positions = []
        scores = []
        for u in range(num_users):
            u_predictions = full_predictions[u * num_items:u * num_items + num_items]
            u_predictions[df_train[df_train[0] == u][1].to_list()] = -np.inf
            top_k_id = u_predictions.argsort()[-topk:][::-1]
            positions.append(np.array(top_k_id))
            top_k_score = u_predictions[top_k_id]
            scores.append(np.array(top_k_score))
            for i, value in enumerate(top_k_id):
                out.write(str(u) + '\t' + str(value) + '\t' + str(top_k_score[i]) + '\n')
                if u % 100 == 0:
                    print("\t Pred. computed for {0}/{1}".format(u + 1, num_users))

        write.save_obj(positions, result_directory + '_top{0}_pos_ep{1}_LFM'.format(args.topk, args.epoch))
        write.save_obj(scores, result_directory + '_top{0}_score_ep{1}_LFM'.format(args.topk, args.epoch))

