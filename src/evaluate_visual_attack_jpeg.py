from cnn.evaluate.visual_metrics import *
from utils.read import *
from cnn.models.dataset import *
import argparse
import csv
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Run visual metrics evaluation.")
    parser.add_argument('--dataset', type=str, default='amazon_men', help='dataset path: amazon_men, amazon_women')
    parser.add_argument('--attack_type', nargs='?', type=str, default='fgsm')
    parser.add_argument('--origin_class', type=int, default=806)
    parser.add_argument('--target_class', type=int, default=770)
    parser.add_argument('--num_classes', type=int, default=1000)
    parser.add_argument('--gpu', type=int, default=0)

    # attacks specific parameters
    parser.add_argument('--eps', type=float, default=16.0)
    parser.add_argument('--it', type=int, default=1)
    parser.add_argument('--l', type=str, default='inf')
    parser.add_argument('--confidence', type=int, default=0)
    parser.add_argument('--nb_iter', type=int, default=10)
    parser.add_argument('--c', type=float, default=0)

    return parser.parse_args()


def evaluate_visual_metrics_jpeg():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    path_all_origin_images, path_original_features, \
    path_attacked_images, path_attacked_classes, path_attacked_features, \
    path_output_visual_metrics_csv = read_config(
        sections_fields=[('PATHS', 'InputImages'),
                         ('PATHS', 'OutputFeatures'),
                         ('PATHS', 'OutputImagesAttack'),
                         ('PATHS', 'OutputClassesAttack'),
                         ('PATHS', 'OutputFeaturesAttack'),
                         ('PATHS', 'OutputVisualMetrics')])

    path_all_origin_images = path_all_origin_images.format(args.dataset)

    df_attacked_classification = None

    if args.attack_type == 'fgsm':
        path_output_visual_metrics_csv = path_output_visual_metrics_csv.format(str(args.dataset),
                                                                               str(args.attack_type),
                                                                               str(args.origin_class),
                                                                               str(args.target_class),
                                                                               'eps' + str(args.eps),
                                                                               'it' + str(args.it),
                                                                               'l' + str(args.l),
                                                                               'XX')

        path_attacked_images = path_attacked_images.format(str(args.dataset),
                                                           str(args.attack_type),
                                                           str(args.origin_class),
                                                           str(args.target_class),
                                                           'eps' + str(args.eps),
                                                           'it' + str(args.it),
                                                           'l' + str(args.l),
                                                           'XX')

        path_attacked_features = path_attacked_features.format(args.dataset,
                                                               args.attack_type,
                                                               args.origin_class,
                                                               args.target_class,
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

    elif args.attack_type == 'pgd':
        path_output_visual_metrics_csv = path_output_visual_metrics_csv.format(str(args.dataset),
                                                                               str(args.attack_type),
                                                                               str(args.origin_class),
                                                                               str(args.target_class),
                                                                               'eps' + str(args.eps),
                                                                               'eps_it' + str(args.eps / 255 / 6),
                                                                               'nb_it' + str(args.nb_iter),
                                                                               'l' + str(args.l))

        path_attacked_images = path_attacked_images.format(str(args.dataset),
                                                           str(args.attack_type),
                                                           str(args.origin_class),
                                                           str(args.target_class),
                                                           'eps' + str(args.eps),
                                                           'eps_it' + str(args.eps / 255 / 6),
                                                           'nb_it' + str(args.nb_iter),
                                                           'l' + str(args.l))

        path_attacked_features = path_attacked_features.format(args.dataset,
                                                               args.attack_type,
                                                               args.origin_class,
                                                               args.target_class,
                                                               'eps' + str(args.eps),
                                                               'eps_it' + str(args.eps / 255 / 6),
                                                               'nb_it' + str(args.nb_iter),
                                                               'l' + str(args.l))

        df_attacked_classification = read_csv(path_attacked_classes.format(args.dataset,
                                                                           args.attack_type,
                                                                           args.origin_class,
                                                                           args.target_class,
                                                                           'eps' + str(args.eps),
                                                                           'eps_it' + str(args.eps / 255 / 6),
                                                                           'nb_it' + str(args.nb_iter),
                                                                           'l' + str(args.l)))

    origin_data = CustomDataset(root_dir=path_all_origin_images)
    attacked_data = CustomDataset(root_dir=path_attacked_images)

    origin_data.filenames = attacked_data.filenames
    origin_data.num_samples = attacked_data.num_samples

    original_features = read_np(filename=path_original_features.format(args.dataset))
    attacked_features = read_np(filename=path_attacked_features.format(args.dataset))

    attacked_indices = [int(os.path.splitext(idx)[0]) for idx in attacked_data.filenames]
    original_features = original_features[attacked_indices]
    attacked_features = attacked_features[attacked_indices]

    print('Starting visual metrics evaluation...')

    avg_mse = 0
    avg_psnr = 0
    avg_ssim = 0
    avg_perceptual_loss = 0
    num_total_attacked = attacked_data.num_samples
    num_targeted_attacked = 0
    num_untargeted_attacked = 0
    num_non_attacked = 0

    with open(path_output_visual_metrics_csv, 'w') as f:
        fieldnames = ['ImageID', 'Mse', 'Psnr', 'Ssim', 'Percep']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i, (d, a) in enumerate(zip(origin_data, attacked_data)):
            origin_img, origin_filename = d
            attacked_img, attacked_filename = a

            # Normalize images between [0, 1]
            origin_img = np.array(origin_img / np.float32(255), dtype=np.float32)
            attacked_img = np.array(attacked_img / np.float32(255), dtype=np.float32)

            # Calculate current visual metrics
            current_mse = mse(im1=origin_img, im2=attacked_img)
            current_psnr = psnr(im1=origin_img, im2=attacked_img)
            current_ssim = ssim(im1=origin_img, im2=attacked_img)
            current_perceptual_loss = mse(im1=original_features[i, :], im2=attacked_features[i, :])

            # Accumulate visual metrics to calculate the final average on attacked images
            avg_mse += current_mse
            avg_psnr += current_psnr
            avg_ssim += current_ssim
            avg_perceptual_loss += current_perceptual_loss

            writer.writerow({'ImageID': os.path.splitext(origin_filename)[0],
                             'Mse': current_mse,
                             'Psnr': current_psnr,
                             'Ssim': current_ssim,
                             'Percep': current_perceptual_loss
                             })

            if df_attacked_classification.loc[df_attacked_classification["ImageID"] == int(
                    os.path.splitext(origin_filename)[0]), "ClassNum"].item() == args.target_class:
                num_targeted_attacked += 1

            elif df_attacked_classification.loc[df_attacked_classification["ImageID"] == int(
                    os.path.splitext(origin_filename)[0]), "ClassNum"].item() == args.origin_class:
                num_non_attacked += 1
            else:
                num_untargeted_attacked += 1

            if (i + 1) % 100 == 0:
                print('%d/%d samples completed' % (i + 1, origin_data.num_samples))

    with open(os.path.dirname(path_output_visual_metrics_csv) + '/average_results', mode='w') as f:
        print('Visual evaluation completed.', file=f)
        print('Final visual metrics:', file=f)
        print('- Average mse: %.20f' % (avg_mse / num_total_attacked), file=f)
        print('- Average psnr: %.20f' % (avg_psnr / num_total_attacked), file=f)
        print('- Average ssim: %.20f' % (avg_ssim / num_total_attacked), file=f)
        print('- Average perceptual loss: %.20f' % (avg_perceptual_loss / num_total_attacked), file=f)
        print('Final success metrics:', file=f)
        print('- Correctly attacked: %.7f' % (num_targeted_attacked / num_total_attacked), file=f)
        print('- Incorrectly attacked: %.7f' % (num_untargeted_attacked / num_total_attacked), file=f)
        print('- Non attacked: %.7f' % (num_non_attacked / num_total_attacked), file=f)


if __name__ == '__main__':
    evaluate_visual_metrics_jpeg()
