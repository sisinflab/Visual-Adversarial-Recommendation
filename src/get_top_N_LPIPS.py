from cnn.visual_attack.utils import *
from utils.read import *
from utils.write import *
import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Run SR and LNorm evaluation.")
    parser.add_argument('--num_classes', type=int, default=1000)
    parser.add_argument('--top_n', type=int, default=50)
    parser.add_argument('--attack_type', nargs='?', type=str, default='fgsm')
    parser.add_argument('--origin_class', type=int, default=806)
    parser.add_argument('--target_class', type=int, default=770)
    parser.add_argument('--dataset', nargs='?', default='amazon_men',
                        help='dataset path: amazon_men, amazon_women, tradesy')
    parser.add_argument('--defense', type=int, default=0)  # 0 --> no defense mode, 1 --> defense mode
    parser.add_argument('--model_dir', type=str, default='free_adv')

    # attacks specific parameters
    parser.add_argument('--eps', type=float, default=16.0)
    parser.add_argument('--it', type=int, default=1)
    parser.add_argument('--l', type=str, default='inf')
    parser.add_argument('--confidence', type=int, default=0)
    parser.add_argument('--nb_iter', type=int, default=100)
    parser.add_argument('--c', type=float, default=0)
    parser.add_argument('--batch_size', type=int, default=128)

    return parser.parse_args()


def get_top_n_lpips():
    args = parse_args()

    ##################################################################################
    # SET PATHS
    # attack + defense || attack
    if args.defense:
        path_images_attacked, path_features_attacked, path_classes_attacked = read_config(
            sections_fields=[('DEFENSE', 'ImagesAttack'),
                             ('DEFENSE', 'FeaturesAttack'),
                             ('DEFENSE', 'ClassesAttack')])
    else:
        path_images_attacked, path_features_attacked, path_classes_attacked = read_config(
            sections_fields=[('ATTACK', 'Images'),
                             ('ATTACK', 'Features'),
                             ('ATTACK', 'Classes')])

    # setting parameters
    params, path_images_attacked, path_classes_attacked, path_features_attacked = set_attack_paths(
        args=args,
        path_images_attack=path_images_attacked,
        path_classes_attack=path_classes_attacked,
        path_features_attack=path_features_attacked
    )
    ##################################################################################
    # READ LPIPS CSV AND WRITE TOP 50

    df = read_csv('../lpips_results/' + args.dataset + '/'
                  + os.path.basename(os.path.dirname(path_classes_attacked)) + '/lpips.csv')

    top_n = df.sort_values(by=['LPIPS']).iloc[:args.top_n]
    write_csv(top_n, '../lpips_results/' + args.dataset + '/'
              + os.path.basename(os.path.dirname(path_classes_attacked)) + '/top_{0}_lpips.csv'.format(str(args.top_n)))
    ##################################################################################


if __name__ == '__main__':
    get_top_n_lpips()
