from PIL import Image
import numpy as np
import pickle
import os


def write_csv(df, filename):
    """
    Args:
        df: pandas dataframe to write
        filename (str): path to store the dataframe
    """
    if not os.path.exists(os.path.dirname(filename)):
        print('\n\nDirectory path %s does not exist. Creating it...' % filename)
        os.makedirs(os.path.dirname(filename))

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
    if not os.path.exists(os.path.dirname(filename)):
        print('\n\nDirectory path %s does not exist. Creating it...' % filename)
        os.makedirs(os.path.dirname(filename))

    np.save(filename, npy)


def save_image(image, filename):
    """
    Store an image to hard disk
    Args:
        image (pytorch tensor): image to save
        filename (str): filename
    """
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))

    im = Image.fromarray(image)
    im.save(filename)

