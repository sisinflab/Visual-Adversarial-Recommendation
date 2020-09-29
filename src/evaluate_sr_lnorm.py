from cnn.visual_attack.utils import *
from torchvision import transforms
from cnn.models.dataset import *
from utils.read import *
import argparse
import csv


def parse_args():
    parser = argparse.ArgumentParser(description="Run SR and LNorm evaluation.")
    parser.add_argument('--num_classes', type=int, default=1000)
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


def eval_sr_ln():
    args = parse_args()

    ##################################################################################
    # LOAD IMAGES
    # original images
    path_images_original, path_features_original, path_classes_original = read_config(
        sections_fields=[('ORIGINAL', 'Images'),
                         ('ORIGINAL', 'Features'),
                         ('ORIGINAL', 'Classes')])
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

    # setting original parameters
    params, path_images_original, path_classes_original, path_features_original = set_attack_paths(
        args=args,
        path_images_attack=path_images_original,
        path_classes_attack=path_classes_original,
        path_features_attack=path_features_original
    )
    # setting attacked parameters
    params, path_images_attacked, path_classes_attacked, path_features_attacked = set_attack_paths(
        args=args,
        path_images_attack=path_images_attacked,
        path_classes_attack=path_classes_attacked,
        path_features_attack=path_features_attacked
    )

    # loading the two datasets
    to_tensor = transforms.ToTensor()
    original_data = CustomDataset(root_dir=path_images_original,
                                  transform=transforms.Compose([to_tensor]))

    attack_data = CustomDataset(root_dir=path_images_attacked,
                                transform=transforms.Compose([to_tensor]))

    original_data.filenames = [os.path.splitext(file)[0] + '.jpg' for file in attack_data.filenames]
    original_data.num_samples = attack_data.num_samples
    ##################################################################################

    ##################################################################################
    # LOAD CSV FILES AND SET OUTPUT FILES
    df = read_csv(path_classes_attacked)
    num_attacked = len(df)

    csv_out = '../visual_results/' + args.dataset + '/' + \
              os.path.basename(os.path.dirname(path_classes_attacked)) + '/lnorm.csv'
    txt_out = '../visual_results/' + args.dataset + '/' + \
              os.path.basename(os.path.dirname(path_classes_attacked)) + '/final_sr_lnorm.txt'

    if not os.path.exists(csv_out):
        os.makedirs(os.path.dirname(csv_out))

    ##################################################################################
    # EVALUATE METRICS
    correct_attack = 0
    lnorm = 0.0

    with open(csv_out, 'w') as f:
        fieldnames = ['ImageID', 'LNorm']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i, (o, a) in enumerate(zip(original_data, attack_data)):
            img0, name0 = o
            img1, name1 = a

            img0 = img0.permute(1, 2, 0).detach().cpu().numpy()
            img1 = img1.permute(1, 2, 0).detach().cpu().numpy()

            current_lnorm = calculate_norm(
                im=(img0 - img1).reshape(-1),
                norm_type=str(args.l)) / calculate_norm(im=img0.reshape(-1),
                                                        norm_type=str(args.l))
            lnorm += current_lnorm
            writer.writerow({
                'ImageID': os.path.splitext(name0)[0],
                'LNorm': current_lnorm
            })

            if df.iloc[i]["ClassNum"] == args.target_class:
                correct_attack += 1

    print('End of metrics calculation.')
    print('Original images read from: %s' % path_images_original)
    print('Attacked/Defended images read from: %s' % path_images_attacked)
    print('\nCALCULATED METRICS:')
    print('\t - Number of correctly attacked samples: %d/%d' % (correct_attack, attack_data.num_samples))
    print('\t - SR (percent): %.5f' % ((correct_attack / num_attacked) * 100))
    print('\t - Average Normalized L-dissimilarity: %.10f' % (lnorm / num_attacked))

    with open(txt_out, "w") as f:
        print('End of metrics calculation.', file=f)
        print('Original images read from: %s' % path_images_original, file=f)
        print('Attacked/Defended images read from: %s' % path_images_attacked, file=f)
        print('\nCALCULATED METRICS:', file=f)
        print('\t - Number of correctly attacked samples: %d/%d' % (correct_attack, attack_data.num_samples), file=f)
        print('\t - SR (percent): %.5f' % ((correct_attack / num_attacked) * 100), file=f)
        print('\t - Average Normalized L-dissimilarity: %.10f' % (lnorm / num_attacked), file=f)


if __name__ == '__main__':
    eval_sr_ln()
