from cnn.evaluate.visual_metrics import *
from cnn.visual_attack.attack import *
from cnn.visual_attack.utils import *
from cnn.models.model import *
from utils.read import *
from utils.write import *
from cnn.models.dataset import *
import torchvision.models as models
from torchvision import transforms
import argparse
import csv
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Run visual metrics evaluation.")
    parser.add_argument('--dataset', type=str, default='amazon_men', help='dataset path: amazon_men, amazon_women')
    parser.add_argument('--attack_type', nargs='?', type=str, default='fgsm')
    parser.add_argument('--origin_class', type=int, default=409)
    parser.add_argument('--target_class', type=str, default=530)
    parser.add_argument('--num_classes', type=int, default=1000)
    parser.add_argument('--gpu', type=int, default=0)

    # attacks specific parameters
    parser.add_argument('--eps', type=float, default=8)
    parser.add_argument('--it', type=int, default=1)
    parser.add_argument('--l', type=str, default='inf')
    parser.add_argument('--confidence', type=int, default=0)
    parser.add_argument('--nb_iter', type=int, default=10)
    parser.add_argument('--c', type=float, default=0)

    return parser.parse_args()


def parse_ord(ord_str):
    if ord_str == 'inf':
        return np.inf
    else:
        return int(ord_str)


def evaluate_visual_metrics():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    path_all_origin_images, path_original_classes, path_original_features, \
    path_attacked_classes, path_attacked_features, path_output_visual_metrics_csv, \
    path_output_images_numpy = read_config(
        sections_fields=[('PATHS', 'InputImages'),
                         ('PATHS', 'OutputClasses'),
                         ('PATHS', 'OutputFeatures'),
                         ('PATHS', 'OutputClassesAttack'),
                         ('PATHS', 'OutputFeaturesAttack'),
                         ('PATHS', 'OutputVisualMetrics'),
                         ('PATHS', 'OutputImagesAttackLossless')])

    path_all_origin_images = path_all_origin_images.format(args.dataset)
    df_origin_classification = read_csv(path_original_classes.format(args.dataset))

    params = None
    df_attacked_classification = None

    if args.attack_type == 'fgsm':
        params = {
            "eps": args.eps / 255,  #
            "clip_min": None,
            "clip_max": None,
            "ord": parse_ord(args.l),  #
            "y_target": None
        }
        path_output_visual_metrics_csv = path_output_visual_metrics_csv.format(str(args.dataset),
                                                                               str(args.attack_type),
                                                                               str(args.origin_class),
                                                                               str(args.target_class),
                                                                               'eps' + str(args.eps),
                                                                               'it' + str(args.it),
                                                                               'l' + str(args.l),
                                                                               'XX')

        path_output_images_numpy = path_output_images_numpy.format(str(args.dataset),
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
        path_output_visual_metrics_csv = path_output_visual_metrics_csv.format(str(args.dataset),
                                                                               str(args.attack_type),
                                                                               str(args.origin_class),
                                                                               str(args.target_class),
                                                                               'eps' + str(args.eps),
                                                                               'eps_it' + str(args.eps / 255 / 6),
                                                                               'nb_it' + str(args.nb_iter),
                                                                               'l' + str(args.l))

        path_output_images_numpy = path_output_images_numpy.format(str(args.dataset),
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
                                                               'eps_it' + str(params["eps_iter"]),
                                                               'nb_it' + str(params["nb_iter"]),
                                                               'l' + str(params["ord"]))

        df_attacked_classification = read_csv(path_attacked_classes.format(args.dataset,
                                                                           args.attack_type,
                                                                           args.origin_class,
                                                                           args.target_class,
                                                                           'eps' + str(args.eps),
                                                                           'eps_it' + str(params["eps_iter"]),
                                                                           'nb_it' + str(params["nb_iter"]),
                                                                           'l' + str(params["ord"])))

    origin_data = CustomDataset(root_dir=path_all_origin_images,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                         std=[0.229, 0.224, 0.225])
                                ])
                                )
    model = Model(model=models.resnet50(pretrained=True))
    attack = VisualAttack(df_classes=df_origin_classification,
                          origin_class=args.origin_class,
                          target_class=args.target_class,
                          model=model.model,
                          device=model.device,
                          params=params,
                          attack_type=args.attack_type,
                          num_classes=args.num_classes)

    original_features = read_np(filename=path_original_features.format(args.dataset))
    attacked_features = read_np(filename=path_attacked_features.format(args.dataset))

    denormalize = transforms.Normalize(mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                                       std=[1 / 0.229, 1 / 0.224, 1 / 0.225])

    print('Starting visual metrics evaluation...')

    avg_mse = 0
    avg_psnr = 0
    avg_ssim = 0
    avg_perceptual_loss = 0
    avg_norm = 0
    num_total_attacked = 0
    num_targeted_attacked = 0
    num_untargeted_attacked = 0
    num_non_attacked = 0

    with open(path_output_visual_metrics_csv, 'w') as f:
        fieldnames = ['ImageID', 'Mse', 'Psnr', 'Ssim', 'Percep', 'Norm']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i, d in enumerate(origin_data):
            origin_img, origin_filename = d

            if attack.must_attack(filename=origin_filename):
                # Generate attacked image
                attacked_img = attack.run_attack(image=origin_img[None, ...])

                # Denormalize original image and transform into numpy
                origin_img = denormalize(origin_img)
                origin_img = origin_img.permute(1, 2, 0).detach().cpu().numpy()

                # Denormalize attacked image, clip to 0.0 and 1.0 and transform into numpy
                attacked_img = denormalize(attacked_img[0])
                attacked_img[attacked_img < 0.0] = 0.0
                attacked_img[attacked_img > 1.0] = 1.0
                attacked_img = attacked_img.permute(1, 2, 0).detach().cpu().numpy()

                # Calculate current visual metrics
                current_mse = mse(im1=origin_img, im2=attacked_img)
                current_psnr = psnr(im1=origin_img, im2=attacked_img)
                current_ssim = ssim(im1=origin_img, im2=attacked_img)
                current_perceptual_loss = mse(im1=original_features[i, :], im2=attacked_features[i, :])
                current_norm = calculate_norm(im1=origin_img.reshape(3, -1),
                                              im2=attacked_img.reshape(3, -1),
                                              norm_type=str(args.l))

                num_total_attacked += 1
                save_image(image=attacked_img,
                           filename=path_output_images_numpy + os.path.splitext(origin_filename)[0],
                           mode='lossless')

                if df_attacked_classification.loc[df_attacked_classification["ImageID"] == int(
                        os.path.splitext(origin_filename)[0]), "ClassNum"].item() == args.target_class:
                    # Calculate the visual metrics only if the targeted attack was successful.
                    # Accumulate visual metrics to calculate the final average on attacked images
                    avg_mse += current_mse
                    avg_psnr += current_psnr
                    avg_ssim += current_ssim
                    avg_perceptual_loss += current_perceptual_loss
                    avg_norm += current_norm
                    num_targeted_attacked += 1

                    writer.writerow({'ImageID': origin_filename,
                                     'Mse': current_mse,
                                     'Psnr': current_psnr,
                                     'Ssim': current_ssim,
                                     'Percep': current_perceptual_loss,
                                     'Norm': current_norm})

                elif df_attacked_classification.loc[df_attacked_classification["ImageID"] == int(
                        os.path.splitext(origin_filename)[0]), "ClassNum"].item() == args.origin_class:
                    num_non_attacked += 1
                else:
                    num_untargeted_attacked += 1

            if (i + 1) % 100 == 0:
                print('%d/%d samples completed' % (i + 1, origin_data.num_samples))

    print('Visual evaluation completed. Final results:')
    print('- Average mse: %.7f' % (avg_mse / num_targeted_attacked))
    print('- Average psnr: %.5f' % (avg_psnr / num_targeted_attacked))
    print('- Average ssim: %.7f' % (avg_ssim / num_targeted_attacked))
    print('- Average perceptual loss: %.7f' % (avg_perceptual_loss / num_targeted_attacked))
    print('- Average norm: %.7f' % (avg_norm / num_targeted_attacked))
    print('- Correctly attacked: %.7f' % (num_targeted_attacked / num_total_attacked))
    print('- Incorrectly attacked: %.7f' % (num_untargeted_attacked / num_total_attacked))
    print('- Non attacked: %.7f' % (num_non_attacked / num_total_attacked))


if __name__ == '__main__':
    evaluate_visual_metrics()
