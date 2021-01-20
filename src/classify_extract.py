from cnn.models.dataset import *
from cnn.models.model import *
from utils.read import *
from utils.write import *
from torchvision import transforms
import torchvision.models as models
import pandas as pd
import numpy as np
import argparse
import time
import sys
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Run classification and feature extraction for original images.")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--dataset', nargs='?', default='tradesy',
                        help='dataset path: amazon_men, amazon_women, tradesy')
    parser.add_argument('--defense', type=int, default=0)  # 0 --> no defense mode, 1 --> defense mode
    parser.add_argument('--model_dir', type=str, default='free_adv')
    parser.add_argument('--model_file', type=str, default='model_best.pth.tar')
    parser.add_argument('--drop_layers', type=int, default=2, help='layers to drop for feature model')
    parser.add_argument('--extract', type=bool, default=True, help='whether to extract features or not')
    parser.add_argument('--resize', type=int, default=224,
                        help='0 --> no resize, otherwise resize to (resize, resize)')
    parser.add_argument('--separate_outputs', type=bool, default=False,
                        help='whether to store (or not) feature numpy separately')
    parser.add_argument('--add_args_features', type=str, default='',
                        help='additional arguments to add to features path (ACF for ACF features)')

    return parser.parse_args()


def classify_and_extract():
    args = parse_args()

    #########################################################################################################
    # MODEL SETTING

    if args.defense:
        path_images, path_output_classes, path_output_features, \
            path_output_features_dir, path_classes, model_path = read_config(
                sections_fields=[('ORIGINAL', 'Images'),
                                 ('DEFENSE', 'Classes'),
                                 ('DEFENSE', 'Features'),
                                 ('DEFENSE', 'FeaturesDir'),
                                 ('ALL', 'ImagenetClasses'),
                                 ('ALL', 'ModelPath')])

        path_output_classes, path_output_features, path_output_features_dir = \
            path_output_classes.format(args.dataset, args.model_dir), \
            path_output_features.format(args.dataset, args.model_dir, args.add_args_features), \
            path_output_features_dir.format(args.dataset, args.model_dir)

        model_path = model_path.format(args.model_dir, args.model_file)
        model = Model(model=models.resnet50(), gpu=args.gpu, model_path=model_path, pretrained_name=args.model_dir)

    else:
        path_images, path_output_classes, path_output_features, \
            path_output_features_dir, path_classes, model_path = read_config(
                sections_fields=[('ORIGINAL', 'Images'),
                                 ('ORIGINAL', 'Classes'),
                                 ('ORIGINAL', 'Features'),
                                 ('ORIGINAL', 'FeaturesDir'),
                                 ('ALL', 'ImagenetClasses'),
                                 ('ALL', 'ModelPath')])
        path_output_classes, path_output_features, path_output_features_dir = \
            path_output_classes.format(args.dataset), \
            path_output_features.format(args.dataset, args.add_args_features), \
            path_output_features_dir.format(args.dataset)

        model = Model(model=models.resnet50(pretrained=True), gpu=args.gpu, model_name='ResNet50')

    if args.extract:
        if not os.path.exists(path_output_features_dir):
            os.makedirs(path_output_features_dir)

        model.set_out_layer(drop_layers=args.drop_layers)
    #########################################################################################################
    #
    #########################################################################################################
    # DATASET SETTING

    path_images = path_images.format(args.dataset)

    data = CustomDataset(root_dir=path_images,
                         transform=transforms.Compose(
                             ([transforms.Resize((args.resize, args.resize),
                                                 interpolation=Image.BICUBIC)] if args.resize else []) + \
                              [transforms.ToTensor()] + \
                              [transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])]
                         ))

    print('Loaded dataset from %s' % path_images)
    #########################################################################################################

    #########################################################################################################
    # READ IMAGENET CLASS NAMES, SET DATAFRAME AND FEATURES

    img_classes = read_imagenet_classes_txt(path_classes)

    df = pd.DataFrame([], columns={'ImageID', 'ClassStr', 'ClassNum', 'Prob'})

    if args.extract:
        if not args.separate_outputs:
            features = np.empty(shape=(data.num_samples, *model.output_shape))
    #########################################################################################################

    #########################################################################################################
    # CLASSIFICATION AND FEATURE EXTRACTION

    print('Starting classification...\n')
    start = time.time()

    for i, d in enumerate(data):
        out_class = model.classification(list_classes=img_classes, sample=d)

        if args.extract:
            if not args.separate_outputs:
                features[i] = model.feature_extraction(sample=d)
            else:
                cnn_features = model.feature_extraction(sample=d)
                cnn_features = cnn_features.reshape((1,
                                                     cnn_features.shape[1],
                                                     cnn_features.shape[2],
                                                     cnn_features.shape[0]))
                save_np(npy=cnn_features,
                        filename=path_output_features_dir + str(i) + '.npy')
        df = df.append(out_class, ignore_index=True)

        if (i + 1) % 100 == 0:
            sys.stdout.write('\r%d/%d samples completed' % (i + 1, data.num_samples))
            sys.stdout.flush()

    write_csv(df=df, filename=path_output_classes)

    if args.extract:
        if not args.separate_outputs:
            save_np(npy=features, filename=path_output_features)

    end = time.time()

    print('\n\nClassification and feature extraction completed in %f seconds.' % (end - start))

    if args.extract:
        if not args.separate_outputs:
            print('Saved features numpy in ==> %s' % path_output_features)
        else:
            print('Saved features numpy in ==> %s' % path_output_features_dir)
    print('Saved classification file in ==> %s' % path_output_classes)
    #########################################################################################################


if __name__ == '__main__':
    classify_and_extract()

