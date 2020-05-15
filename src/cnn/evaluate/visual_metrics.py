from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity
import numpy as np


def mse(im1, im2):
    return mean_squared_error(im1, im2)


def rmse(im1, im2):
    return np.sqrt(mse(im1, im2))


def psnr(im1, im2):
    return peak_signal_noise_ratio(im1, im2)


def ssim(im1, im2):
    return structural_similarity(im1, im2, multichannel=True)

