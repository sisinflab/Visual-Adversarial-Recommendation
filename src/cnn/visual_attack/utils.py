import numpy as np


def calculate_norm(im1, im2, norm_type):
    if norm_type in ['0', '1', '2']:
        return np.linalg.norm(im1 - im2, ord=int(norm_type), axis=1)
    elif norm_type == 'inf':
        return np.linalg.norm(im1 - im2, ord=np.inf, axis=1)
