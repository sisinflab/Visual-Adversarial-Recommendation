import tensorflow as tf

def mse(im1, im2):
    return tf.math.reduce_mean(tf.math.squared_difference(im1, im2)).numpy()

def psnr(im1, im2, max_val):
    return tf.image.psnr(im1, im2, max_val=max_val).numpy()

def ssim(im1, im2, max_val):
    return tf.image.ssim(im1, im2, max_val=max_val).numpy()