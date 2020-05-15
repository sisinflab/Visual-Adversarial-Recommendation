from cnn.evaluate.visual_metrics import *
from cnn.visual_attack.utils import *
from utils.read import *
import argparse
import csv
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Run feature loss evaluation.")
    parser.add_argument('--num_classes', type=int, default=1000)
    parser.add_argument('--attack_type', nargs='?', type=str, default='fgsm')
    parser.add_argument('--origin_class', type=int, default=774)
    parser.add_argument('--target_class', type=int, default=770)
    parser.add_argument('--dataset', nargs='?', default='amazon_men',
                        help='dataset path: amazon_men, amazon_women, tradesy')
    parser.add_argument('--defense', type=int, default=0)  # 0 --> no defense mode, 1 --> defense mode
    parser.add_argument('--model_dir', type=str, default='free_adv')

    # attacks specific parameters
    parser.add_argument('--eps', type=float, default=4.0)
    parser.add_argument('--it', type=int, default=1)
    parser.add_argument('--l', type=str, default='inf')
    parser.add_argument('--confidence', type=int, default=0)
    parser.add_argument('--nb_iter', type=int, default=100)
    parser.add_argument('--c', type=float, default=0)
    parser.add_argument('--batch_size', type=int, default=128)

    return parser.parse_args()


def evaluate_feature_loss():
    args = parse_args()

    if args.defense:
        path_input_images_attack, path_input_features_attack, path_input_classes_attack, \
            path_input_features = read_config(
                sections_fields=[('DEFENSE', 'ImagesAttack'),
                                 ('DEFENSE', 'FeaturesAttack'),
                                 ('DEFENSE', 'ClassesAttack'),
                                 ('DEFENSE', 'Features')])
        path_input_features = path_input_features.format(args.dataset, args.model_dir)

    else:
        path_input_images_attack, path_input_features_attack, path_input_classes_attack, \
            path_input_features = read_config(
                sections_fields=[('ATTACK', 'Images'),
                                 ('ATTACK', 'Features'),
                                 ('ATTACK', 'Classes'),
                                 ('ORIGINAL', 'Features')])

        path_input_features = path_input_features.format(args.dataset)

    params, path_input_images_attack, path_input_classes_attack, path_input_features_attack = set_attack_paths(
        args=args,
        path_images_attack=path_input_images_attack,
        path_classes_attack=path_input_classes_attack,
        path_features_attack=path_input_features_attack
    )

    original_features = read_np(filename=path_input_features)
    attacked_features = read_np(filename=path_input_features_attack)

    avg_mse_features_loss = 0.0
    avg_rmse_features_loss = 0.0

    df_attacked_classification = read_csv(path_input_classes_attack)

    num_attacked = len(df_attacked_classification)

    output_txt = os.path.dirname(path_input_classes_attack) + '/features_dist_avg_all_attack.txt'
    output_csv = os.path.dirname(path_input_classes_attack) + '/features_dist_all_attack.csv'

    with open(output_csv, 'w') as f:
        fieldnames = ['ImageID', 'MSE', 'RMSE']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for index, row in df_attacked_classification.iterrows():

            current_mse_features_loss = mse(im1=original_features[int(row["ImageID"]), :],
                                            im2=attacked_features[int(row["ImageID"]), :])
            current_rmse_features_loss = rmse(im1=original_features[int(row["ImageID"]), :],
                                              im2=attacked_features[int(row["ImageID"]), :])

            writer.writerow({
                'ImageID': row['ImageID'],
                'MSE': current_mse_features_loss,
                'RMSE': current_rmse_features_loss
            })

            avg_mse_features_loss += current_mse_features_loss
            avg_rmse_features_loss += current_rmse_features_loss

    with open(output_txt, 'w') as f:
        print('Total attacked images: %d' % num_attacked, file=f)
        print('Final MSE features loss: %.8f' % (avg_mse_features_loss / num_attacked), file=f)
        print('Final RMSE features loss: %.8f' % (avg_rmse_features_loss / num_attacked), file=f)


if __name__ == '__main__':
    evaluate_feature_loss()

