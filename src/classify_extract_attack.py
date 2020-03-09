from cnn.models.dataset import *
from cnn.models.model import *
from cnn.visual_attack.attack import *
from utils.read import *
from utils.write import *
from torchvision import transforms
import torchvision.models as models
import pandas as pd
import argparse
import sys
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Run classification and feature extraction for a specific attack.")
    parser.add_argument('--num_classes', type=int, default=1000)
    parser.add_argument('--attack_type', nargs='?', type=str, default='fgsm')
    parser.add_argument('--origin_class', type=int, default=531)
    parser.add_argument('--target_class', type=int, default=770)
    parser.add_argument('--eps', type=float, default=4)
    parser.add_argument('--clip_min', type=float, default=0)
    parser.add_argument('--clip_max', type=float, default=1)
    parser.add_argument('--it', type=int, default=1)
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
    df_origin_classification = read_csv(path_input_classes)
    params = {'eps': args.eps / 255,
              'clip_min': args.clip_min,
              'clip_max': args.clip_max
             }
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
                          params=params,
                          attack_type=args.attack_type,
                          num_classes=args.num_classes)
    imgnet_classes = read_imagenet_classes_txt(path_classes)

    df = pd.DataFrame([], columns={'ImageID', 'ClassNumStart', 'ClassStrStart', 'ClassNum', 'ClassStr'})

    features = read_np(filename=path_input_features)

    for i, d in enumerate(data):
        im, name = d

        if attack.must_attack(filename=name):
            if args.attack_type == 'fgsm':
                attacked = attack.run_fgsm(image=im)
                save_image(image=attacked, filename=path_output_images_attack.format(args.attack_type,
                                                                                     args.origin_class,
                                                                                     args.target_class,
                                                                                     args.eps,
                                                                                     args.it) + name)
            elif args.attack_type == 'pgd':
                attacked = attack.run_pgd(image=im)
                save_image(image=attacked, filename=path_output_images_attack.format(args.attack_type,
                                                                                     args.origin_class,
                                                                                     args.target_class,
                                                                                     args.eps,
                                                                                     args.it) + name)
            elif args.attack_type == 'c_w':
                attacked = attack.run_c_w(image=im)
                save_image(image=attacked, filename=path_output_images_attack.format(args.attack_type,
                                                                                     args.origin_class,
                                                                                     args.target_class,
                                                                                     args.eps,
                                                                                     args.it) + name)
            elif args.attack_type == 'deep_fool':
                attacked = attack.run_deep_fool(image=im)
                save_image(image=attacked, filename=path_output_images_attack.format(args.attack_type,
                                                                                     args.origin_class,
                                                                                     args.target_class,
                                                                                     args.eps,
                                                                                     args.it) + name)
            else:
                attacked = None
                print('Attack type not known. Available attack types: [fgsm, pgd, c_w, deep_fool]')
                exit(0)

            out_class = model.classification(list_classes=imgnet_classes, sample=(attacked, name))
            features[i, :] = model.feature_extraction(sample=(attacked, name))
            out_class["ClassStrStart"] = df_origin_classification.loc[df_origin_classification["ImageID"] == int(os.path.splitext(name)[0]), "ClassStr"].item()
            out_class["ClassNumStart"] = df_origin_classification.loc[df_origin_classification["ImageID"] == int(os.path.splitext(name)[0]), "ClassStr"].item()
            df = df.append(out_class, ignore_index=True)

        sys.stdout.write('\r%d/%d samples completed' % (i + 1, data.num_samples))
        sys.stdout.flush()

    write_csv(df=df, filename=path_output_classes_attack.format(args.attack_type,
                                                                args.origin_class,
                                                                args.target_class,
                                                                args.eps,
                                                                args.it))
    save_np(npy=features, filename=path_output_features_attack.format(args.attack_type,
                                                                      args.origin_class,
                                                                      args.target_class,
                                                                      args.eps,
                                                                      args.it))

if __name__ == '__main__':
    classify_and_extract_attack()
