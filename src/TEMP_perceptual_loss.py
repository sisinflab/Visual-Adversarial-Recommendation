from cnn.evaluate.visual_metrics import *
from cnn.visual_attack.utils import *
from utils.read import *
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Run attack evaluation.")
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


def evaluate_perceptual_loss():
    args = parse_args()

    if args.defense:
        path_input_images_attack, path_input_features_attack, path_input_classes_attack = read_config(
            sections_fields=[('DEFENSE', 'ImagesAttack'),
                             ('DEFENSE', 'FeaturesAttack'),
                             ('DEFENSE', 'ClassesAttack')])

    else:
        path_input_images_attack, path_input_features_attack, path_input_classes_attack = read_config(
            sections_fields=[('ATTACK', 'Images'),
                             ('ATTACK', 'Features'),
                             ('ATTACK', 'Classes')])

    params, path_input_images_attack, path_input_classes_attack, path_input_features_attack = set_attack_paths(
        args=args,
        path_images_attack=path_input_images_attack,
        path_classes_attack=path_input_classes_attack,
        path_features_attack=path_input_features_attack
    )

    original_features = read_np(filename=path_input_features_attack)
    attacked_features = read_np(filename=path_input_features_attack)

    avg_perceptual_loss = 0.0
    num_targeted_attacked = 0

    df_attacked_classification = read_csv(path_input_classes_attack)

    for index, row in df_attacked_classification.iterrows():

        if row["ClassNum"] == args.target_class:
            current_perceptual_loss = mse(im1=original_features[int(row["ImageID"]), :],
                                          im2=attacked_features[int(row["ImageID"]), :])
            avg_perceptual_loss += current_perceptual_loss
            num_targeted_attacked += 1

        if (index + 1) % 100 == 0:
            print('%d/%d samples completed' % (index + 1, len(df_attacked_classification)))

    print('\n\nFinal perceptual loss for %s attack: %.8f' % (args.attack_type,
                                                             avg_perceptual_loss / num_targeted_attacked))


if __name__ == '__main__':
    evaluate_perceptual_loss()
