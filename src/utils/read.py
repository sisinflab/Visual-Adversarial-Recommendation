import pandas as pd
import configparser
import pickle
import numpy as np
from PIL import Image


def read_csv(filename):
    """
    Args:
        filename (str): csv file path

    Return:
         A pandas dataframe.
    """
    df = pd.read_csv(filename, index_col=False)
    return df


def read_np(filename):
    """
    Args:
        filename (str): filename of numpy to load
    Return:
        The loaded numpy.
    """
    return np.load(filename)


def read_imagenet_classes_txt(filename):
    """
    Args:
        filename (str): txt file path

    Return:
         A list with 1000 imagenet classes as strings.
    """
    with open(filename) as f:
        idx2label = eval(f.read())

    return idx2label


def read_config(sections_fields):
    """
    Args:
        sections_fields (list): list of fields to retrieve from configuration file

    Return:
         A list of configuration values.
    """
    config = configparser.ConfigParser()
    config.read('./../config/configs.ini')
    configs = []
    for s, f in sections_fields:
        configs.append(config[s][f])
    return configs


def read_image(filename):
    return Image.open(filename)


def load_obj(name):
    """
    Load the pkl object by name
    :param name: name of file
    :return:
    """
    with open(name, 'rb') as f:
        return pickle.load(f)

