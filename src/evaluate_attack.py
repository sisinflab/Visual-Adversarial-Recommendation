from utils.read import *
from cnn.visual_attack.utils import *
import argparse
import os

attacks_params = {
    "fgsm": {
        "name": "Fast Gradient Sign Method (FGSM)"
    },
    "cw": {
        "name": "Carlini & Wagner (C & W)"
    },
    "pgd": {
        "name": "Projected Gradient Descent (PGD)"
    },
    "jsma": {
        "name": "Jacobian-based Saliency Map Attack (JSMA)"
    },
    "zoo": {
        "name": "Zeroth Order Optimization (ZOO)"
    },
    "spsa": {
        "name": "Simultaneous Perturbation Stochastic Approximation (SPSA)"
    }

}


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


def evaluate_attack():
    args = parse_args()

    if args.defense:
        path_input_images_attack, path_input_features_attack, path_input_classes_attack, path_classes = read_config(
            sections_fields=[('DEFENSE', 'ImagesAttack'),
                             ('DEFENSE', 'FeaturesAttack'),
                             ('DEFENSE', 'ClassesAttack'),
                             ('ALL', 'ImagenetClasses')])

    else:
        path_input_images_attack, path_input_features_attack, path_input_classes_attack, path_classes = read_config(
            sections_fields=[('ATTACK', 'Images'),
                             ('ATTACK', 'Features'),
                             ('ATTACK', 'Classes'),
                             ('ALL', 'ImagenetClasses')])

    params, path_input_images_attack, path_input_classes_attack, path_input_features_attack = set_attack_paths(
        args=args,
        path_images_attack=path_input_images_attack,
        path_classes_attack=path_input_classes_attack,
        path_features_attack=path_input_features_attack
    )

    imgnet_classes = read_imagenet_classes_txt(path_classes)

    df = read_csv(path_input_classes_attack)

    out_file = os.path.dirname(path_input_classes_attack) + '/success_results.txt'

    with open(out_file, "w") as f:
        print("EVALUATING {0} ATTACK on DATASET {1}".format(attacks_params[args.attack_type]["name"], args.dataset),
              file=f)
        print("- ORIGINAL CLASS: %d/%d (%s)" % (args.origin_class, args.num_classes - 1,
                                                imgnet_classes[args.origin_class]), file=f)
        print("- TARGET CLASS: %d/%d (%s)" % (args.target_class, args.num_classes - 1,
                                              imgnet_classes[args.target_class]), file=f)
        print("- PARAMETERS:", file=f)
        for key in params:
            print("\t- " + key + " = " + str(params[key]), file=f)

        print("\n\nRESULTS:", file=f)
        print("\t- TARGETED ATTACK SUCCESS RATE: %f" % ((len(df[df["ClassNum"] == args.target_class]) / len(df)) * 100),
              file=f)


if __name__ == '__main__':
    evaluate_attack()

