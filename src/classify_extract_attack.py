from cnn.models.dataset import *
from cnn.models.model import *
from cnn.visual_attack.attack import *
from utils.read import *
from utils.write import *
from torchvision import transforms
import torchvision.models as models
import numpy as np
import argparse
import csv
import os
import shutil

# per parametri vuoti, usare X

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
    }

}


def parse_ord(ord_str):
    if ord_str == 'inf':
        return np.inf
    else:
        return int(ord_str)


def parse_args():
    parser = argparse.ArgumentParser(description="Run classification and feature extraction for a specific attack.")
    parser.add_argument('--num_classes', type=int, default=1000)
    parser.add_argument('--attack_type', nargs='?', type=str, default='fgsm')
    parser.add_argument('--origin_class', type=int, default=409)
    parser.add_argument('--target_class', type=int, default=530)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--dataset', nargs='?', default='amazon_men',
                        help='dataset path: amazon_men, amazon_women')

    # attacks specific parameters
    parser.add_argument('--eps', type=float, default=8)
    parser.add_argument('--it', type=int, default=1)
    parser.add_argument('--l', type=str, default='inf')
    parser.add_argument('--confidence', type=int, default=0)
    parser.add_argument('--nb_iter', type=int, default=10)
    parser.add_argument('--c', type=float, default=0)

    return parser.parse_args()


def classify_and_extract_attack():
    args = parse_args()

    path_images, path_input_classes, path_input_features, path_classes, \
    path_output_images_attack, path_output_features_attack, path_output_classes_attack = read_config(
        sections_fields=[('PATHS', 'InputImages'),
                         ('PATHS', 'OutputClasses'),
                         ('PATHS', 'OutputFeatures'),
                         ('PATHS', 'ImagenetClasses'),
                         ('PATHS', 'OutputImagesAttack'),
                         ('PATHS', 'OutputFeaturesAttack'),
                         ('PATHS', 'OutputClassesAttack')])

    path_images, path_input_classes, path_input_features = path_images.format(args.dataset), path_input_classes.format(
        args.dataset), path_input_features.format(args.dataset)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    if args.attack_type == 'fgsm':
        params = {
            "eps": args.eps / 255,  #
            "clip_min": None,
            "clip_max": None,
            "ord": parse_ord(args.l),  #
            "y_target": None
        }
        path_output_images_attack = path_output_images_attack.format(args.dataset,
                                                                     args.attack_type,
                                                                     args.origin_class,
                                                                     args.target_class,
                                                                     'eps' + str(args.eps),
                                                                     'it' + str(args.it),
                                                                     'l' + str(args.l),
                                                                     'XX')
        path_output_classes_attack = path_output_classes_attack.format(args.dataset,
                                                                       args.attack_type,
                                                                       args.origin_class,
                                                                       args.target_class,
                                                                       'eps' + str(args.eps),
                                                                       'it' + str(args.it),
                                                                       'l' + str(args.l),
                                                                       'XX')
        path_output_features_attack = path_output_features_attack.format(args.dataset,
                                                                         args.attack_type,
                                                                         args.origin_class,
                                                                         args.target_class,
                                                                         'eps' + str(args.eps),
                                                                         'it' + str(args.it),
                                                                         'l' + str(args.l),
                                                                         'XX')

    elif args.attack_type == 'pgd':
        params = {
            "eps": args.eps / 255,
            "eps_iter": args.eps / 255 / 6,  #
            "nb_iter": 10,  #
            "ord": parse_ord(args.l),  #
            "clip_min": None,
            "clip_max": None,
            "y_target": None,
            "rand_init": None,
            "rand_init_eps": None,
            "clip_grad": False,
            "sanity_checks": True
        }
        path_output_images_attack = path_output_images_attack.format(args.dataset,
                                                                     args.attack_type,
                                                                     args.origin_class,
                                                                     args.target_class,
                                                                     'eps' + str(args.eps),
                                                                     'eps_it' + str(args.eps / 255 / 6),
                                                                     'nb_it' + str(params["nb_iter"]),
                                                                     'l' + str(params["ord"]))
        path_output_classes_attack = path_output_classes_attack.format(args.dataset,
                                                                       args.attack_type,
                                                                       args.origin_class,
                                                                       args.target_class,
                                                                       'eps' + str(args.eps),
                                                                       'eps_it' + str(params["eps_iter"]),
                                                                       'nb_it' + str(params["nb_iter"]),
                                                                       'l' + str(params["ord"]))
        path_output_features_attack = path_output_features_attack.format(args.dataset,
                                                                         args.attack_type,
                                                                         args.origin_class,
                                                                         args.target_class,
                                                                         'eps' + str(args.eps),
                                                                         'eps_it' + str(params["eps_iter"]),
                                                                         'nb_it' + str(params["nb_iter"]),
                                                                         'l' + str(params["ord"]))

    # TO REVISE Carlini & Wagner and JSMA
    elif args.attack_type == 'cw':
        params = {
            "y_target": None,
            "batch_size": 1,
            "confidence": args.confidence,  #
            "learning_rate": 5e-3,
            "binary_search_steps": 0,  #
            "max_iterations": 1000,  #
            "abort_early": True,
            "initial_const": args.c,  #
            "clip_min": args.clip_min,
            "clip_max": args.clip_min
        }
        path_output_images_attack = path_output_images_attack.format(args.dataset,
                                                                     args.attack_type,
                                                                     args.origin_class,
                                                                     args.target_class,
                                                                     'conf' + str(params["confidence"]),
                                                                     'lr' + str(params["learning_rate"]),
                                                                     'c' + str(params["initial_const"]),
                                                                     'max_it' + str(params["max_iterations"]))
        path_output_classes_attack = path_output_classes_attack.format(args.dataset,
                                                                       args.attack_type,
                                                                       args.origin_class,
                                                                       args.target_class,
                                                                       'conf' + str(params["confidence"]),
                                                                       'lr' + str(params["learning_rate"]),
                                                                       'c' + str(params["initial_const"]),
                                                                       'max_it' + str(params["max_iterations"]))
        path_output_features_attack = path_output_features_attack.format(args.dataset,
                                                                         args.attack_type,
                                                                         args.origin_class,
                                                                         args.target_class,
                                                                         'conf' + str(params["confidence"]),
                                                                         'lr' + str(params["learning_rate"]),
                                                                         'c' + str(params["initial_const"]),
                                                                         'max_it' + str(params["max_iterations"]))

    elif args.attack_type == 'jsma':
        params = {
            "theta": 1.0,  #
            "gamma": 1.0,  #
            "clip_min": args.clip_min,
            "clip_max": args.clip_max,
            "y_target": None,
            "symbolic_impl": True  #
        }
        path_output_images_attack = path_output_images_attack.format(args.dataset,
                                                                     args.attack_type,
                                                                     args.origin_class,
                                                                     args.target_class,
                                                                     'th' + str(params["theta"]),
                                                                     'ga' + str(params["gamma"]),
                                                                     'symb' + str(params["symbolic_impl"]),
                                                                     'XX')
        path_output_classes_attack = path_output_classes_attack.format(args.dataset,
                                                                       args.attack_type,
                                                                       args.origin_class,
                                                                       args.target_class,
                                                                       'th' + str(params["theta"]),
                                                                       'ga' + str(params["gamma"]),
                                                                       'symb' + str(params["symbolic_impl"]),
                                                                       'XX')
        path_output_features_attack = path_output_features_attack.format(args.dataset,
                                                                         args.attack_type,
                                                                         args.origin_class,
                                                                         args.target_class,
                                                                         'th' + str(params["theta"]),
                                                                         'ga' + str(params["gamma"]),
                                                                         'symb' + str(params["symbolic_impl"]),
                                                                         'XX')
    else:
        print('Unknown attack type.')
        exit(0)

    imgnet_classes = read_imagenet_classes_txt(path_classes)

    print("RUNNING {0} ATTACK on DATASET {1}".format(attacks_params[args.attack_type]["name"], args.dataset))
    print("- ORIGINAL CLASS: %d/%d (%s)" % (args.origin_class, args.num_classes - 1, imgnet_classes[args.origin_class]))
    print("- TARGET CLASS: %d/%d (%s)" % (args.target_class, args.num_classes - 1, imgnet_classes[args.target_class]))
    print("- PARAMETERS:")
    for key in params:
        print("\t- " + key + " = " + str(params[key]))
    print("\n")

    df_origin_classification = read_csv(path_input_classes)
    data = CustomDataset(root_dir=path_images,
                         transform=transforms.Compose([
                             transforms.ToTensor(),
                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                  std=[0.229, 0.224, 0.225])
                         ]))
    model = Model(model=models.resnet50(pretrained=True))
    model.set_out_layer(drop_layers=1)
    attack = VisualAttack(df_classes=df_origin_classification,
                          origin_class=args.origin_class,
                          target_class=args.target_class,
                          model=model.model,
                          device=model.device,
                          params=params,
                          attack_type=args.attack_type,
                          num_classes=args.num_classes)

    features = read_np(filename=path_input_features)

    denormalize = transforms.Normalize(mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                                       std=[1 / 0.229, 1 / 0.224, 1 / 0.225])

    # Check if attack directory already created. If so, delete it, and then create it again.
    if os.path.exists(os.path.dirname(path_output_classes_attack)):
        shutil.rmtree(os.path.dirname(path_output_classes_attack))
    os.makedirs(os.path.dirname(path_output_classes_attack))

    with open(path_output_classes_attack, 'w') as f:
        fieldnames = ['ImageID', 'ClassNum', 'ClassStr', 'Prob', 'ClassNumStart', 'ClassStrStart', 'ProbStart']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i, d in enumerate(data):
            im, name = d

            if attack.must_attack(filename=name):
                # Generate attacked image with chosen attack algorithm
                adv_perturbed_out = attack.run_attack(image=im[None, ...])

                # Classify attacked image with pretrained model and append new classification to csv
                out_class = model.classification(list_classes=imgnet_classes,
                                                 sample=(adv_perturbed_out[0], name))
                out_class["ClassStrStart"] = df_origin_classification.loc[
                    df_origin_classification["ImageID"] == int(os.path.splitext(name)[0]), "ClassStr"].item()
                out_class["ClassNumStart"] = df_origin_classification.loc[
                    df_origin_classification["ImageID"] == int(os.path.splitext(name)[0]), "ClassNum"].item()
                out_class["ProbStart"] = df_origin_classification.loc[
                    df_origin_classification["ImageID"] == int(os.path.splitext(name)[0]), "Prob"].item()
                writer.writerow(out_class)

                # Extract features using pretrained model
                features[i, :] = model.feature_extraction(sample=(adv_perturbed_out[0], name))

                # Denormalize image before saving to memory
                adv_perturbed_out = denormalize(adv_perturbed_out[0])

                # Clip before saving image to memory
                adv_perturbed_out[adv_perturbed_out < 0.0] = 0.0
                adv_perturbed_out[adv_perturbed_out > 1.0] = 1.0

                # Save image to memory
                save_image(image=adv_perturbed_out, filename=path_output_images_attack + name)

            if (i + 1) % 1000 == 0:
                print('%d/%d samples completed' % (i + 1, data.num_samples))

    # Save all extracted features (attacked and non-attacked ones)
    save_np(npy=features, filename=path_output_features_attack)


if __name__ == '__main__':
    classify_and_extract_attack()
