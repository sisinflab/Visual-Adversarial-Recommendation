from cnn.models.dataset import *
from cnn.models.model import *
from cnn.visual_attack.attack import *
from cnn.visual_attack.utils import set_attack_paths
from utils.read import *
from utils.write import *
from torchvision import transforms
import torchvision.models as models
import tensorflow as tf
import torch
import numpy as np
import argparse
import csv
import os
import time
import shutil
import random
import sys

# set random seed to make reproducible experiments
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
tf.compat.v1.set_random_seed(0)

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
    parser = argparse.ArgumentParser(description="Run classification and feature extraction for a specific attack.")
    parser.add_argument('--num_classes', type=int, default=1000)
    parser.add_argument('--attack_type', nargs='?', type=str, default='pgd')
    parser.add_argument('--origin_class', type=int, default=774)
    parser.add_argument('--target_class', type=int, default=770)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--dataset', nargs='?', default='amazon_men',
                        help='dataset path: amazon_men, amazon_women, amazon_beauty')
    parser.add_argument('--defense', type=int, default=0)  # 0 --> no defense mode, 1 --> defense mode
    parser.add_argument('--model_dir', type=str, default='free_adv')
    parser.add_argument('--model_file', type=str, default='model_best.pth.tar')

    # attacks specific parameters
    parser.add_argument('--eps', type=float, default=8)
    parser.add_argument('--it', type=int, default=1)
    parser.add_argument('--l', type=str, default='inf')
    parser.add_argument('--confidence', type=int, default=0)
    parser.add_argument('--nb_iter', type=int, default=100)
    parser.add_argument('--c', type=float, default=0)
    parser.add_argument('--batch_size', type=int, default=128)

    return parser.parse_args()


def classify_and_extract_attack():
    args = parse_args()

    #########################################################################################################
    # MODEL SETTING

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    if args.defense:
        path_images, path_input_classes, path_input_features, path_classes, model_path, \
            path_output_images_attack, path_output_features_attack, path_output_classes_attack = read_config(
                sections_fields=[('ORIGINAL', 'Images'),
                                 ('ORIGINAL', 'Classes'),
                                 ('ORIGINAL', 'Features'),
                                 ('ALL', 'ImagenetClasses'),
                                 ('ALL', 'ModelPath'),
                                 ('DEFENSE', 'ImagesAttack'),
                                 ('DEFENSE', 'FeaturesAttack'),
                                 ('DEFENSE', 'ClassesAttack')])

        path_input_classes, path_input_features = path_input_classes.format(
            args.dataset, args.model_dir), path_input_features.format(args.dataset, args.model_dir)

        model_path = model_path.format(args.model_dir, args.model_file)
        model = Model(model=models.resnet50(), model_path=model_path, pretrained_name=args.model_dir)

    else:
        path_images, path_input_classes, path_input_features, path_classes, model_path, \
            path_output_images_attack, path_output_features_attack, path_output_classes_attack = read_config(
                sections_fields=[('ORIGINAL', 'Images'),
                                 ('ORIGINAL', 'Classes'),
                                 ('ORIGINAL', 'Features'),
                                 ('ALL', 'ImagenetClasses'),
                                 ('ALL', 'ModelPath'),
                                 ('ATTACK', 'Images'),
                                 ('ATTACK', 'Features'),
                                 ('ATTACK', 'Classes')])

        path_input_classes, path_input_features = path_input_classes.format(
            args.dataset), path_input_features.format(args.dataset)

        model = Model(model=models.resnet50(pretrained=True), model_name='ResNet50')

    model.set_out_layer(drop_layers=1)
    #########################################################################################################

    #########################################################################################################
    # DATASET SETTING

    path_images = path_images.format(args.dataset)

    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    denormalize = transforms.Normalize(mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                                       std=[1 / 0.229, 1 / 0.224, 1 / 0.225])

    # Resize images for black-box attacks which are computationally expensive
    if args.attack_type in ['zoo']:  # HO TOLTO SPSA
        data = CustomDataset(root_dir=path_images,
                             reshape=True,
                             scale=4,
                             transform=transforms.Compose([
                                 to_tensor,
                                 normalize
                             ]))

    else:
        data = CustomDataset(root_dir=path_images,
                             transform=transforms.Compose([
                                 to_tensor,
                                 normalize
                             ]))

    print('Loaded dataset from %s' % path_images)
    #########################################################################################################

    #########################################################################################################
    # PATHS SETTING, IMAGENET CLASSES LOADING, ORIGINAL CLASSIFICATION LOADING AND ORIGINAL FEATURES LOADING

    params, path_output_images_attack, path_output_classes_attack, path_output_features_attack = set_attack_paths(
        args=args,
        path_images_attack=path_output_images_attack,
        path_classes_attack=path_output_classes_attack,
        path_features_attack=path_output_features_attack
    )

    imgnet_classes = read_imagenet_classes_txt(path_classes)

    df_origin_classification = read_csv(path_input_classes)

    features = read_np(filename=path_input_features)
    #########################################################################################################

    #########################################################################################################
    # VISUAL ATTACK SETTING

    print("\nRUNNING {0} ATTACK on DATASET {1}".format(attacks_params[args.attack_type]["name"], args.dataset))
    print("- ORIGINAL CLASS: %d/%d (%s)" % (args.origin_class, args.num_classes - 1, imgnet_classes[args.origin_class]))
    print("- TARGET CLASS: %d/%d (%s)" % (args.target_class, args.num_classes - 1, imgnet_classes[args.target_class]))
    print("- PARAMETERS:")
    for key in params:
        print("\t- " + key + " = " + str(params[key]))

    attack = VisualAttack(df_classes=df_origin_classification,
                          tf_pytorch='tf' if args.attack_type in ['cw', 'jsma', 'zoo', 'spsa'] else 'pytorch',
                          origin_class=args.origin_class,
                          target_class=args.target_class,
                          model=model.model,
                          device=model.device,
                          params=params,
                          attack_type=args.attack_type,
                          num_classes=args.num_classes)

    #########################################################################################################

    #########################################################################################################
    # ATTACK GENERATION, CLASSIFICATION AND FEATURE EXTRACTION

    print('Starting attack generation, classification and feature extraction...\n')
    start = time.time()

    total_attack_time = 0.0

    if not os.path.exists(os.path.dirname(path_output_classes_attack)):
        os.makedirs(os.path.dirname(path_output_classes_attack))
    else:
        shutil.rmtree(os.path.dirname(path_output_classes_attack))
        os.makedirs(os.path.dirname(path_output_classes_attack))

    with open(path_output_classes_attack, 'w') as f:
        fieldnames = ['ImageID',
                      'ClassNum', 'ClassStr', 'Prob',
                      'ClassNumStart', 'ClassStrStart', 'ProbStart']

        # if args.attack_type in ['spsa', 'zoo']:
        #     fieldnames = ['ImageID',
        #                   'ClassNum', 'ClassStr', 'Prob',  # new classification on small images
        #                   'RClassNum', 'RClassStr', 'RProb',  # new classification on resized images
        #                   'ClassNumStart', 'ClassStrStart', 'ProbStart']  # old classification
        # else:
        #     fieldnames = ['ImageID',
        #                   'ClassNum', 'ClassStr', 'Prob',
        #                   'ClassNumStart', 'ClassStrStart', 'ProbStart']

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i, d in enumerate(data):

            if args.attack_type in ['zoo']:  # HO TOLTO SPSA
                im, height, width, name = d

            else:
                im, name = d

            if attack.must_attack(filename=name):

                # Generate attacked image with chosen attack algorithm
                start_attack = time.time()
                adv_perturbed_out = attack.run_attack(image=im[None, ...])
                end_attack = time.time()
                total_attack_time += (end_attack - start_attack)

                if args.attack_type in ['spsa', 'zoo']:
                    print('\n\n***************%d/%d samples completed***************\n\n' % (i + 1, data.num_samples))

                # Denormalize image before saving to memory
                adv_perturbed_out = denormalize(adv_perturbed_out[0])

                # Clip before saving image to memory
                adv_perturbed_out[adv_perturbed_out < 0.0] = 0.0
                adv_perturbed_out[adv_perturbed_out > 1.0] = 1.0

                # Transform into numpy, permute and multiply by 255 (uint8)
                adv_perturbed_out = (adv_perturbed_out.permute(1, 2, 0).detach().cpu().numpy() * 255).astype('uint8')

                # Save image as tiff (lossless compression)
                save_image(image=adv_perturbed_out,
                           filename=path_output_images_attack + os.path.splitext(name)[0] + '.tiff')

                # Read same image from memory
                lossless_image = read_image(path_output_images_attack + os.path.splitext(name)[0] + '.tiff')

                # If the attack is black-box, perform reshape to get the original image size
                # However, classification and feature extraction will be performed on the downscaled image
                # if args.attack_type in ['spsa', 'zoo']:
                #     # Resize image to original dimension
                #     reshaped_lossless_image = Image.fromarray(adv_perturbed_out).resize(size=(height, width),
                #                                                                         resample=Image.BICUBIC)
                #
                #     # Transform to tensor and normalize
                #     reshaped_lossless_image = normalize(to_tensor(reshaped_lossless_image))
                #
                #     # Classify attacked image with pre-trained model and append new classification to csv
                #     out_class_reshaped = model.classification(list_classes=imgnet_classes,
                #                                               sample=(reshaped_lossless_image, name))

                # Transform to tensor and normalize
                lossless_image = normalize(to_tensor(lossless_image))

                # Classify attacked image with pre-trained model and append new classification to csv
                out_class = model.classification(list_classes=imgnet_classes,
                                                 sample=(lossless_image, name))

                out_class["ClassStrStart"] = df_origin_classification.loc[
                    df_origin_classification["ImageID"] == int(os.path.splitext(name)[0]), "ClassStr"].item()
                out_class["ClassNumStart"] = df_origin_classification.loc[
                    df_origin_classification["ImageID"] == int(os.path.splitext(name)[0]), "ClassNum"].item()
                out_class["ProbStart"] = df_origin_classification.loc[
                    df_origin_classification["ImageID"] == int(os.path.splitext(name)[0]), "Prob"].item()

                # If the attack is black-box, add these other parameters to the output
                # if args.attack_type in ['spsa', 'zoo']:
                #     out_class["RClassNum"] = out_class_reshaped["ClassNum"]
                #     out_class["RClassStr"] = out_class_reshaped["ClassStr"]
                #     out_class["RProb"] = out_class_reshaped["Prob"]

                writer.writerow(out_class)

                # Extract features using pre-trained model
                features[i, :] = model.feature_extraction(sample=(lossless_image, name))

            if (i + 1) % 100 == 0 and args.attack_type not in ['spsa', 'zoo']:
                sys.stdout.write('\r%d/%d samples completed' % (i + 1, data.num_samples))
                sys.stdout.flush()

    # Save all extracted features (attacked and non-attacked ones)
    save_np(npy=features, filename=path_output_features_attack)

    end = time.time()

    print('\n\nAttack, classification and feature extraction completed in %f seconds.' % (end - start))
    print('Attack completed in %f seconds.' % total_attack_time)
    print('Saved features numpy in ==> %s' % path_output_features_attack)
    print('Saved classification file in ==> %s' % path_output_classes_attack)
    #########################################################################################################


if __name__ == '__main__':
    classify_and_extract_attack()

