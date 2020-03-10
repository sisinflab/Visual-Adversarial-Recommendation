from cnn.models.dataset import *
from cnn.models.model import *
from cnn.visual_attack.attack import *
from utils.read import *
from utils.write import *
from torchvision import transforms
import torchvision.models as models
import pandas as pd
import numpy as np
import argparse
import sys
import os

# per parametri vuoti, usare X

attacks_params = {
    "fgsm":{
            "iter": 1, #
            "eps_denorm": 4, #
            "ord_str": 'inf',
            "params":{
                "eps": 4 / 255,
                "clip_min": 0.0,
                "clip_max": 1.0,
                "ord": np.inf, #
                "y_target": None
            }
    },
    "cw":{
            "params":{
                "y_target": None,
                "batch_size": 1,
                "confidence": 0, #
                "learning_rate": 5e-3, #
                "binary_search_steps": 5, #
                "max_iterations": 1000, #
                "abort_early": True,
                "initial_const": 1e-2,
                "clip_min": 0.0,
                "clip_max": 1.0
            }
    },
    "pgd":{
            "ord_str": 'inf',
            "params":{
                "eps": 0.3, #
                "eps_iter": 0.05, #
                "nb_iter": 10, #
                "ord": np.inf, #
                "clip_min": 0.0,
                "clip_max": 1.0,
                "y_target": None,
                "rand_init": None,
                "rand_init_eps": None,
                "clip_grad": False,
                "sanity_checks": True
            }
    },
    "jsma":{
        "params":{
                "theta": 1.0, #
                "gamma": 1.0, #
                "clip_min": 0.0,
                "clip_max": 1.0,
                "y_target": None,
                "symbolic_impl": True #
        }
    }

}

def parse_args():
    parser = argparse.ArgumentParser(description="Run classification and feature extraction for a specific attack.")
    parser.add_argument('--num_classes', type=int, default=1000)
    parser.add_argument('--attack_type', nargs='?', type=str, default='fgsm')
    parser.add_argument('--origin_class', type=int, default=531)
    parser.add_argument('--target_class', type=int, default=770)
    return parser.parse_args()

def classify_and_extract_attack():
    path_images, path_input_classes, path_input_features, path_classes, \
    path_output_images_attack, path_output_features_attack, path_output_classes_attack = read_config(
        sections_fields=[('PATHS', 'InputAmazonMenImages'),
                         ('PATHS', 'OutputAmazonMenClasses'),
                         ('PATHS', 'OutputAmazonMenFeatures'),
                         ('PATHS', 'ImagenetClasses'),
                         ('PATHS', 'OutputAmazonMenImagesAttack'),
                         ('PATHS', 'OutputAmazonMenFeaturesAttack'),
                         ('PATHS', 'OutputAmazonMenClassesAttack')])

    args = parse_args()

    if args.attack_type == 'fgsm':
        path_output_images_attack = path_output_images_attack.format(args.attack_type,
                                                                     args.origin_class,
                                                                     args.target_class,
                                                                     'eps' + str(attacks_params[args.attack_type]["eps_denorm"]),
                                                                     'it' + str(attacks_params[args.attack_type]["iter"]),
                                                                     'l' + str(attacks_params[args.attack_type]["ord_str"]),
                                                                     'XX')
        path_output_classes_attack = path_output_classes_attack.format(args.attack_type,
                                                                       args.origin_class,
                                                                       args.target_class,
                                                                       'eps' + str(attacks_params[args.attack_type]["eps_denorm"]),
                                                                       'it' + str(attacks_params[args.attack_type]["iter"]),
                                                                       'l' + str(attacks_params[args.attack_type]["ord_str"]),
                                                                       'XX')
        path_output_features_attack = path_output_features_attack.format(args.attack_type,
                                                                         args.origin_class,
                                                                         args.target_class,
                                                                         'eps' + str(attacks_params[args.attack_type]["eps_denorm"]),
                                                                         'it' + str(attacks_params[args.attack_type]["iter"]),
                                                                         'l' + str(attacks_params[args.attack_type]["ord_str"]),
                                                                         'XX')

    elif args.attack_type == 'cw':
        path_output_images_attack = path_output_images_attack.format(args.attack_type,
                                                                     args.origin_class,
                                                                     args.target_class,
                                                                     'confidence' + str(attacks_params[args.attack_type]["params"]["confidence"]),
                                                                     'learning_rate' + str(attacks_params[args.attack_type]["params"]["learning_rate"]),
                                                                     'binary_search_steps' + str(attacks_params[args.attack_type]["params"]["binary_search_steps"]),
                                                                     'max_iterations' + str(attacks_params[args.attack_type]["params"]["max_iterations"]))
        path_output_classes_attack = path_output_classes_attack.format(args.attack_type,
                                                                       args.origin_class,
                                                                       args.target_class,
                                                                       'confidence' + str(attacks_params[args.attack_type]["params"]["confidence"]),
                                                                       'learning_rate' + str(attacks_params[args.attack_type]["params"]["learning_rate"]),
                                                                       'binary_search_steps' + str(attacks_params[args.attack_type]["params"]["binary_search_steps"]),
                                                                       'max_iterations' + str(attacks_params[args.attack_type]["params"]["max_iterations"]))
        path_output_features_attack = path_output_features_attack.format(args.attack_type,
                                                                         args.origin_class,
                                                                         args.target_class,
                                                                         'confidence' + str(attacks_params[args.attack_type]["params"]["confidence"]),
                                                                         'learning_rate' + str(attacks_params[args.attack_type]["params"]["learning_rate"]),
                                                                         'binary_search_steps' + str(attacks_params[args.attack_type]["params"]["binary_search_steps"]),
                                                                         'max_iterations' + str(attacks_params[args.attack_type]["params"]["max_iterations"]))

    elif args.attack_type == 'pgd':
        path_output_images_attack = path_output_images_attack.format(args.attack_type,
                                                                     args.origin_class,
                                                                     args.target_class,
                                                                     'eps' + str(attacks_params[args.attack_type]["params"]["eps"]),
                                                                     'eps_iter' + str(attacks_params[args.attack_type]["params"]["eps_iter"]),
                                                                     'nb_iter' + str(attacks_params[args.attack_type]["params"]["nb_iter"]),
                                                                     'l' + str(attacks_params[args.attack_type]["ord_str"]))
        path_output_classes_attack = path_output_classes_attack.format(args.attack_type,
                                                                       args.origin_class,
                                                                       args.target_class,
                                                                       'eps' + str(attacks_params[args.attack_type]["params"]["eps"]),
                                                                       'eps_iter' + str(attacks_params[args.attack_type]["params"]["eps_iter"]),
                                                                       'nb_iter' + str(attacks_params[args.attack_type]["params"]["nb_iter"]),
                                                                       'l' + str(attacks_params[args.attack_type]["ord_str"]))
        path_output_features_attack = path_output_features_attack.format(args.attack_type,
                                                                         args.origin_class,
                                                                         args.target_class,
                                                                         'eps' + str(attacks_params[args.attack_type]["params"]["eps"]),
                                                                         'eps_iter' + str(attacks_params[args.attack_type]["params"]["eps_iter"]),
                                                                         'nb_iter' + str(attacks_params[args.attack_type]["params"]["nb_iter"]),
                                                                         'l' + str(attacks_params[args.attack_type]["ord_str"]))

    elif args.attack_type == 'jsma':
        path_output_images_attack = path_output_images_attack.format(args.attack_type,
                                                                     args.origin_class,
                                                                     args.target_class,
                                                                     'theta' + str(attacks_params[args.attack_type]["params"]["theta"]),
                                                                     'gamma' + str(attacks_params[args.attack_type]["params"]["gamma"]),
                                                                     'symbolic_impl' + str(attacks_params[args.attack_type]["params"]["symbolic_impl"]),
                                                                     'XX')
        path_output_classes_attack = path_output_classes_attack.format(args.attack_type,
                                                                       args.origin_class,
                                                                       args.target_class,
                                                                       'theta' + str(attacks_params[args.attack_type]["params"]["theta"]),
                                                                       'gamma' + str(attacks_params[args.attack_type]["params"]["gamma"]),
                                                                       'symbolic_impl' + str(attacks_params[args.attack_type]["params"]["symbolic_impl"]),
                                                                       'XX')
        path_output_features_attack = path_output_features_attack.format(args.attack_type,
                                                                         args.origin_class,
                                                                         args.target_class,
                                                                         'theta' + str(attacks_params[args.attack_type]["params"]["theta"]),
                                                                         'gamma' + str(attacks_params[args.attack_type]["params"]["gamma"]),
                                                                         'symbolic_impl' + str(attacks_params[args.attack_type]["params"]["symbolic_impl"]),
                                                                         'XX')
    else:
        print('Uknown attack type.')
        exit(0)

    print("\n\n***************************************************************")
    print("Running {0} attack".format(args.attack_type))
    print("***************************************************************\n\n")

    df_origin_classification = read_csv(path_input_classes)
    data = CustomDataset(root_dir=path_images,
                         transform=transforms.Compose([
                             transforms.ToTensor()
                         ]))
    model = Model(model=models.resnet50(pretrained=True))
    model.set_out_layer(drop_layers=1)
    attack = VisualAttack(df_classes=df_origin_classification,
                          origin_class=args.origin_class,
                          target_class=args.target_class,
                          model=model.model,
                          params=attacks_params[args.attack_type]["params"],
                          attack_type=args.attack_type,
                          num_classes=args.num_classes)
    imgnet_classes = read_imagenet_classes_txt(path_classes)

    df = pd.DataFrame([], columns={'ImageID', 'ClassNumStart', 'ClassStrStart', 'ClassNum', 'ClassStr'})
    #ClassNum and ClassStr should be the target class if everything works fine

    features = read_np(filename=path_input_features)

    for i, d in enumerate(data):
        im, name = d

        if attack.must_attack(filename=name):
            attacked = attack.run_attack(image=im)

            if args.attack_type == 'fgsm':
                save_image(image=attacked, filename=path_output_images_attack + name)
            elif args.attack_type == 'cw':
                save_image(image=attacked, filename=path_output_images_attack + name)
            elif args.attack_type == 'pgd':
                save_image(image=attacked, filename=path_output_images_attack + name)
            elif args.attack_type == 'jsma':
                save_image(image=attacked, filename=path_output_images_attack + name)

            out_class = model.classification(list_classes=imgnet_classes, sample=(attacked, name))
            features[i, :] = model.feature_extraction(sample=(attacked, name))
            out_class["ClassStrStart"] = df_origin_classification.loc[df_origin_classification["ImageID"] == int(os.path.splitext(name)[0]), "ClassStr"].item()
            out_class["ClassNumStart"] = df_origin_classification.loc[df_origin_classification["ImageID"] == int(os.path.splitext(name)[0]), "ClassNum"].item()
            df = df.append(out_class, ignore_index=True)

        sys.stdout.write('\r%d/%d samples completed' % (i + 1, data.num_samples))
        sys.stdout.flush()

    write_csv(df=df, filename=path_output_classes_attack)
    save_np(npy=features, filename=path_output_features_attack)

if __name__ == '__main__':
    classify_and_extract_attack()
