from cnn.evaluate.visual_metrics import *
from torchvision import transforms
from utils.read import *
from cnn.models.dataset import *
import torchvision.models as models
import argparse
import csv
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Run visual metrics evaluation.")
    parser.add_argument('--dir', type=str, default='../data/amazon_men/original_images/images/')
    parser.add_argument('--gpu', type=int, default=0)

    return parser.parse_args()

def evaluate_visual_metrics():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    path_all_origin_images, path_output_visual_metrics_csv = read_config(
                        sections_fields=[('PATHS', 'InputAmazonMenImages'),
                                         ('PATHS', 'OutputAmazonMenVisualMetrics')])

    ##### ADD NORMALIZATION FOR PERCEPTUAL LOSS ########
    origin_data = CustomDataset(root_dir=path_all_origin_images,
                                transform=transforms.Compose([
                                    transforms.ToTensor()
                                ]))

    attacked_data = CustomDataset(root_dir=args.dir,
                                  transform=transforms.Compose([
                                        transforms.ToTensor()
                                  ]))

    # Select only attacked images from origin
    origin_data.filenames = attacked_data.filenames
    num_attacked = attacked_data.num_samples

    print('Starting visual metrics evaluation...')

    avg_mse = 0
    avg_psnr = 0
    avg_ssim = 0

    with open(path_output_visual_metrics_csv, 'w') as f:
        fieldnames = ['ImageID', 'MSE', 'PSNR', 'SSIM']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i, (origin_d, attacked_d) in enumerate(zip(origin_data, attacked_data)):
            origin_img, origin_filename = origin_d
            attacked_img, attacked_filename = attacked_d

            current_mse = mse(im1=origin_img, im2=attacked_img)
            current_psnr = psnr(im1=origin_img, im2=attacked_img, max_val=1.0)
            current_ssim = ssim(im1=origin_img, im2=attacked_img, max_val=1.0)

            writer.writerow({'ImageID': attacked_filename,
                             'MSE': current_mse,
                             'PSNR': current_psnr,
                             'SSIM': current_ssim})

            avg_mse += current_mse
            avg_psnr += current_psnr
            avg_ssim += current_ssim

            if (i + 1) % 100 == 0:
                print('%d/%d samples completed' % (i + 1, num_attacked))

    print('Visual evaluation completed. Average results:')
    print('- Average mse: %.7f' %(avg_mse / num_attacked))
    print('- Average psnr: %.5f' %(avg_psnr / num_attacked))
    print('- Average ssim: %.7f' %(avg_ssim / num_attacked))


if __name__ == '__main__':
    evaluate_visual_metrics()