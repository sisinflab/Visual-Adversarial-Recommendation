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
    "fgsm": {
        "name": "Fast Gradient Sign Method (FGSM)"
    },
    "pgd": {
        "name": "Projected Gradient Descent (PGD)"
    }

}

def parse_ord(ord_str):
    if ord_str == 'inf':
        return np.inf
    else:
        return int(ord_str)

def parse_args():
    parser = argparse.ArgumentParser(description="Run classification and feature extraction with attacks.")
    parser.add_argument('--num_classes', type=int, default=1000)
    parser.add_argument('--attack_type', nargs='?', type=str, default='cw')
    parser.add_argument('--origin_class', type=int, default=409)
    parser.add_argument('--target_class', type=int, default=770)
    parser.add_argument('--gpu', type=int, default=0)

    # attacks specific parameters
    parser.add_argument('--eps', type=float, default=4)
    parser.add_argument('--it', type=int, default=1)
    parser.add_argument('--l', type=str, default='inf')
    parser.add_argument('--confidence', type=int, default=0)
    parser.add_argument('--nb_iter', type=int, default=10)
    parser.add_argument('--c', type=float, default=0)

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

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    params_fgsm = {
        "eps": args.eps / 255,  #
        "clip_min": 0.0,
        "clip_max": 1.0,
        "ord": parse_ord(args.l),  #
        "y_target": None
    }
    path_output_images_attack_fgsm = path_output_images_attack.format('fgsm',
                                                                      args.origin_class,
                                                                      args.target_class,
                                                                      'eps' + str(args.eps),
                                                                      'it' + str(args.it),
                                                                      'l' + str(args.l),
                                                                      'XX')
    path_output_classes_attack_fgsm = path_output_classes_attack.format('fgsm',
                                                                        args.origin_class,
                                                                        args.target_class,
                                                                        'eps' + str(args.eps),
                                                                        'it' + str(args.it),
                                                                        'l' + str(args.l),
                                                                        'XX')
    path_output_features_attack_fgsm = path_output_features_attack.format('fgsm',
                                                                          args.origin_class,
                                                                          args.target_class,
                                                                          'eps' + str(args.eps),
                                                                          'it' + str(args.it),
                                                                          'l' + str(args.l),
                                                                          'XX')

    params_pgd = {
        "eps": args.eps / 255,
        "eps_iter": args.eps / 255 / 6,  #
        "nb_iter": 10,  #
        "ord": parse_ord(args.l),  #
        "clip_min": 0.0,
        "clip_max": 1.0,
        "y_target": None,
        "rand_init": None,
        "rand_init_eps": None,
        "clip_grad": False,
        "sanity_checks": True
    }
    path_output_images_attack_pgd = path_output_images_attack.format('pgd',
                                                                     args.origin_class,
                                                                     args.target_class,
                                                                     'eps' + str(args.eps),
                                                                     'eps_it' + str(params_pgd["eps_iter"]),
                                                                     'nb_it' + str(params_pgd["nb_iter"]),
                                                                     'l' + str(params_pgd["ord"]))
    path_output_classes_attack_pgd = path_output_classes_attack.format('pgd',
                                                                       args.origin_class,
                                                                       args.target_class,
                                                                       'eps' + str(args.eps),
                                                                       'eps_it' + str(params_pgd["eps_iter"]),
                                                                       'nb_it' + str(params_pgd["nb_iter"]),
                                                                       'l' + str(params_pgd["ord"]))
    path_output_features_attack_pgd = path_output_features_attack.format('pgd',
                                                                         args.origin_class,
                                                                         args.target_class,
                                                                         'eps' + str(args.eps),
                                                                         'eps_it' + str(params_pgd["eps_iter"]),
                                                                         'nb_it' + str(params_pgd["nb_iter"]),
                                                                         'l' + str(params_pgd["ord"]))

    imgnet_classes = read_imagenet_classes_txt(path_classes)

    print("RUNNING {0} ATTACK".format('fgsm'))
    print("- ORIGINAL CLASS: %d/%d (%s)" % (args.origin_class, args.num_classes - 1, imgnet_classes[args.origin_class]))
    print("- TARGET CLASS: %d/%d (%s)" % (args.target_class, args.num_classes - 1, imgnet_classes[args.target_class]))
    print("- PARAMETERS:")
    for key in params_fgsm:
        print("\t- " + key + " = " + str(params_fgsm[key]))
    print("\n")
    print("RUNNING {0} ATTACK".format('pgd'))
    print("- ORIGINAL CLASS: %d/%d (%s)" % (args.origin_class, args.num_classes - 1, imgnet_classes[args.origin_class]))
    print("- TARGET CLASS: %d/%d (%s)" % (args.target_class, args.num_classes - 1, imgnet_classes[args.target_class]))
    print("- PARAMETERS:")
    for key in params_pgd:
        print("\t- " + key + " = " + str(params_pgd[key]))
    print("\n")

    df_origin_classification = read_csv(path_input_classes)
    data = CustomDataset(root_dir=path_images,
                         transform=transforms.Compose([
                             transforms.ToTensor()
                         ]))
    model = Model(model=models.resnet50(pretrained=True))
    model.set_out_layer(drop_layers=1)

    attack_fgsm = VisualAttack(df_classes=df_origin_classification,
                               origin_class=args.origin_class,
                               target_class=args.target_class,
                               model=model.model,
                               params=params_fgsm,
                               attack_type='fgsm',
                               num_classes=args.num_classes)

    attack_pgd = VisualAttack(df_classes=df_origin_classification,
                              origin_class=args.origin_class,
                              target_class=args.target_class,
                              model=model.model,
                              params=params_pgd,
                              attack_type='pgd',
                              num_classes=args.num_classes)

    df_fgsm = pd.DataFrame([], columns={'ImageID', 'ClassNumStart', 'ClassStrStart', 'ProbStart', 'ClassNum', 'ClassStr', 'Prob'})
    df_pgd = pd.DataFrame([], columns={'ImageID', 'ClassNumStart', 'ClassStrStart', 'ProbStart', 'ClassNum', 'ClassStr', 'Prob'})

    features_fgsm = read_np(filename=path_input_features)
    features_pgd = read_np(filename=path_input_features)

    for i, d in enumerate(data):
        im, name = d

        attacked_fgsm = attack_fgsm.run_attack(image=im)
        attacked_pgd = attack_pgd.run_attack(image=im)
        save_image(image=attacked_fgsm, filename=path_output_images_attack_fgsm + name)
        save_image(image=attack_pgd, filename=path_output_images_attack_pgd + name)

        out_class_fgsm = model.classification(list_classes=imgnet_classes, sample=(attacked_fgsm, name))
        out_class_pgd = model.classification(list_classes=imgnet_classes, sample=(attacked_pgd, name))

        features_fgsm[i, :] = model.feature_extraction(sample=(attacked_fgsm, name))
        features_pgd[i, :] = model.feature_extraction(sample=(attack_pgd, name))

        out_class_fgsm["ClassStrStart"] = df_origin_classification.loc[
            df_origin_classification["ImageID"] == int(os.path.splitext(name)[0]), "ClassStr"].item()
        out_class_fgsm["ClassNumStart"] = df_origin_classification.loc[
            df_origin_classification["ImageID"] == int(os.path.splitext(name)[0]), "ClassNum"].item()
        out_class_fgsm["ProbStart"] = df_origin_classification.loc[
            df_origin_classification["ImageID"] == int(os.path.splitext(name)[0]), "ProbStart"].item()
        out_class_pgd["ClassStrStart"] = df_origin_classification.loc[
            df_origin_classification["ImageID"] == int(os.path.splitext(name)[0]), "ClassStr"].item()
        out_class_pgd["ClassNumStart"] = df_origin_classification.loc[
            df_origin_classification["ImageID"] == int(os.path.splitext(name)[0]), "ClassNum"].item()
        out_class_pgd["ProbStart"] = df_origin_classification.loc[
            df_origin_classification["ImageID"] == int(os.path.splitext(name)[0]), "ProbStart"].item()

        df_fgsm = df_fgsm.append(out_class_fgsm, ignore_index=True)
        df_pgd = df_pgd.append(out_class_pgd, ignore_index=True)

        sys.stdout.write('\r%d/%d samples completed' % (i + 1, data.num_samples))
        sys.stdout.flush()

    write_csv(df=df_fgsm, filename=path_output_classes_attack_fgsm)
    save_np(npy=features_fgsm, filename=path_output_features_attack_fgsm)
    write_csv(df=df_pgd, filename=path_output_classes_attack_pgd)
    save_np(npy=features_pgd, filename=path_output_features_attack_pgd)


if __name__ == '__main__':
    classify_and_extract_attack()
