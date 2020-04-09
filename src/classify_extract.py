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


def parse_args():
    parser = argparse.ArgumentParser(description="Run classification and feature extraction for original images.")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--dataset', nargs='?', default='amazon_men',
                        help='dataset path: amazon_men, amazon_women')
    parser.add_argument('--defense', type=int, default=1)  # 0 --> no defense mode, 1 --> defense mode
    parser.add_argument('--model_dir', type=str, default='madrylab')
    parser.add_argument('--model_file', type=str, default='imagenet_linf_8.pt')

    return parser.parse_args()


def classify_and_extract():
    args = parse_args()

    if args.defense:
        path_images, path_output_classes, path_output_features, path_classes, model_path = read_config(
            sections_fields=[('PATHS', 'InputImages'),
                             ('PATHS', 'OutputClassesDefense'),
                             ('PATHS', 'OutputFeaturesDefense'),
                             ('PATHS', 'ImagenetClasses'),
                             ('PATHS', 'ModelPath')])
        path_output_classes, path_output_features = path_output_classes.format(args.dataset, args.model_dir), \
                                                    path_output_features.format(args.dataset, args.model_dir)

    else:
        path_images, path_output_classes, path_output_features, path_classes, model_path = read_config(
            sections_fields=[('PATHS', 'InputImages'),
                             ('PATHS', 'OutputClasses'),
                             ('PATHS', 'OutputFeatures'),
                             ('PATHS', 'ImagenetClasses'),
                             ('PATHS', 'ModelPath')])
        path_output_classes, path_output_features = path_output_classes.format(args.dataset), \
                                                    path_output_features.format(args.dataset)

    path_images = path_images.format(args.dataset)

    data = CustomDataset(root_dir=path_images,
                         transform=transforms.Compose([
                             transforms.ToTensor(),
                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                  std=[0.229, 0.224, 0.225])
                         ]))

    # if model is read from memory
    if args.model_dir:
        model_path = model_path.format(args.model_dir, args.model_file)
        model = Model(model=models.resnet50(), gpu=args.gpu, model_path=model_path, pretrained_name=args.model_dir)

    # otherwise, use a pytorch pre-trained model
    else:
        model = Model(model=models.resnet50(pretrained=True), gpu=args.gpu, model_name='ResNet50')

    model.set_out_layer(drop_layers=1)
    img_classes = read_imagenet_classes_txt(path_classes)

    df = pd.DataFrame([], columns={'ImageID', 'ClassStr', 'ClassNum', 'Prob'})

    features = np.empty(shape=(data.num_samples, 2048))

    print('Starting classification...\n\n')
    start = time.time()

    for i, d in enumerate(data):
        out_class = model.classification(list_classes=img_classes, sample=d)
        features[i, :] = model.feature_extraction(sample=d)
        df = df.append(out_class, ignore_index=True)

        if (i + 1) % 100 == 0:
            sys.stdout.write('\r%d/%d samples completed' % (i + 1, data.num_samples))
            sys.stdout.flush()

    write_csv(df=df, filename=path_output_classes)
    save_np(npy=features, filename=path_output_features)

    print('\n\nClassification completed in %f seconds.' % (time.time() - start))
    print('Saved features numpy in ==> %s' % path_output_features)
    print('Saved classification file in ==> %s' % path_output_classes)


if __name__ == '__main__':
    classify_and_extract()
