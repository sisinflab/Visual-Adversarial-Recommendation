import torchvision
import numpy as np
import pickle
import os


def write_csv(df, filename):
    """
    Args:
        df: pandas dataframe to write
        filename (str): path to store the dataframe
    """
    df.to_csv(filename, index=False)


def save_obj(obj, name):
    """
    Store the object in a pkl file
    :param obj: python object to be stored
    :param name: file name (Not insert .pkl)
    :return:
    """
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f)


def save_np(npy, filename):
    """
    Store numpy to memory.
    Args:
        npy: numpy to save
        filename (str): filename
    """
    np.save(filename, npy)


def save_image(image, filename, mode='lossless'):
    """
    Store an image to hard disk
    Args:
        image (pytorch tensor): image to save
        filename (str): filename
        mode (str): either lossless or lossy
    """
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))

    if mode == 'lossy':
        torchvision.utils.save_image(image, filename + 'jpg')
    elif mode == 'lossless':
        save_np(npy=image, filename=filename)
