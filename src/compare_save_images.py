from matplotlib import pyplot as plt
from utils.read import *
import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Run image comparison and saving.")
    parser.add_argument('--dataset', type=str, default='amazon_men', help='dataset path: amazon_men, amazon_women')
    parser.add_argument('--attack_type', nargs='?', type=str, default='fgsm')
    parser.add_argument('--origin_class', type=int, default=806)
    parser.add_argument('--target_class', type=int, default=770)
    parser.add_argument('--image_id', type=int, default=31)

    # attacks specific parameters
    parser.add_argument('--eps', type=float, default=16.0)
    parser.add_argument('--it', type=int, default=1)
    parser.add_argument('--l', type=str, default='inf')
    parser.add_argument('--confidence', type=int, default=0)
    parser.add_argument('--nb_iter', type=int, default=10)
    parser.add_argument('--c', type=float, default=0)

    return parser.parse_args()


def compare_and_save_images():
    args = parse_args()

    path_all_origin_images, path_attacked_images, path_attacked_classes, path_visual_metrics = read_config(
        sections_fields=[('PATHS', 'InputImages'),
                         ('PATHS', 'OutputImagesAttack'),
                         ('PATHS', 'OutputClassesAttack'),
                         ('PATHS', 'OutputVisualMetrics')])

    path_all_origin_images = path_all_origin_images.format(args.dataset)
    df_attacked_classification = None
    df_attacked_metrics = None

    if args.attack_type == 'fgsm':
        path_attacked_images = path_attacked_images.format(str(args.dataset),
                                                           str(args.attack_type),
                                                           str(args.origin_class),
                                                           str(args.target_class),
                                                           'eps' + str(args.eps),
                                                           'it' + str(args.it),
                                                           'l' + str(args.l),
                                                           'XX')
        df_attacked_classification = read_csv(path_attacked_classes.format(args.dataset,
                                                                           args.attack_type,
                                                                           args.origin_class,
                                                                           args.target_class,
                                                                           'eps' + str(args.eps),
                                                                           'it' + str(args.it),
                                                                           'l' + str(args.l),
                                                                           'XX'))

        df_attacked_metrics = read_csv(path_visual_metrics.format(args.dataset,
                                                                  args.attack_type,
                                                                  args.origin_class,
                                                                  args.target_class,
                                                                  'eps' + str(args.eps),
                                                                  'it' + str(args.it),
                                                                  'l' + str(args.l),
                                                                  'XX'))

    elif args.attack_type == 'pgd':
        path_attacked_images = path_attacked_images.format(str(args.dataset),
                                                           str(args.attack_type),
                                                           str(args.origin_class),
                                                           str(args.target_class),
                                                           'eps' + str(args.eps),
                                                           'eps_it' + str(args.eps / 255 / 6),
                                                           'nb_it' + str(args.nb_iter),
                                                           'l' + str(args.l))

        df_attacked_classification = read_csv(path_attacked_classes.format(str(args.dataset),
                                                                           str(args.attack_type),
                                                                           str(args.origin_class),
                                                                           str(args.target_class),
                                                                           'eps' + str(args.eps),
                                                                           'eps_it' + str(args.eps / 255 / 6),
                                                                           'nb_it' + str(args.nb_iter),
                                                                           'l' + str(args.l)))

        df_attacked_metrics = read_csv(path_visual_metrics.format(str(args.dataset),
                                                                  str(args.attack_type),
                                                                  str(args.origin_class),
                                                                  str(args.target_class),
                                                                  'eps' + str(args.eps),
                                                                  'eps_it' + str(args.eps / 255 / 6),
                                                                  'nb_it' + str(args.nb_iter),
                                                                  'l' + str(args.l)))

    original_class = df_attacked_classification.loc[df_attacked_classification['ImageID'] == args.image_id, 'ClassStrStart'].item()
    original_prob = df_attacked_classification.loc[df_attacked_classification['ImageID'] == args.image_id, 'ProbStart'].item()
    attacked_class = df_attacked_classification.loc[df_attacked_classification['ImageID'] == args.image_id, 'ClassStr'].item()
    attacked_prob = df_attacked_classification.loc[df_attacked_classification['ImageID'] == args.image_id, 'Prob'].item()

    mse = df_attacked_metrics.loc[df_attacked_metrics['ImageID'] == args.image_id, 'Mse'].item()
    psnr = df_attacked_metrics.loc[df_attacked_metrics['ImageID'] == args.image_id, 'Psnr'].item()
    ssim = df_attacked_metrics.loc[df_attacked_metrics['ImageID'] == args.image_id, 'Ssim'].item()
    percep = df_attacked_metrics.loc[df_attacked_metrics['ImageID'] == args.image_id, 'Percep'].item()

    print('Image: %s' % str(args.image_id) + '.jpg')
    print('Original class: %s' % original_class)
    print('Original probability: %.20f' % original_prob)
    print('Attacked class: %s' % attacked_class)
    print('Attacked probability: %.20f' % attacked_prob)
    print('MSE: %.20f' % mse)
    print('PSNR: %.20f' % psnr)
    print('SSIM: %.20f' % ssim)
    print('PL: %.20f' % percep)
    origin_class = plt.imread(path_all_origin_images + str(args.image_id) + '.jpg')
    attacked_class = plt.imread(path_attacked_images + str(args.image_id) + '.jpg')
    plt.imsave('../data/pdf_images/' + str(args.image_id) + '_original.pdf', origin_class)
    plt.imsave('../data/pdf_images/' + str(args.image_id) + '_attacked.pdf',
               attacked_class)


if __name__ == '__main__':
    compare_and_save_images()

