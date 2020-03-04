import pandas as pd
import configparser
import pickle

def read_csv(filename):
    """
    Args:
        filename (str): csv file path

    Return:
         A pandas dataframe.
    """
    df = pd.read_csv(filename)
    return df

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

def read_config(config_file, sections_fields):
    """
    Args:
        config_file (str): configuration file
        sections_fields (list): list of fields to retrieve from configuration file

    Return:
         A list of configuration values.
    """
    config = configparser.ConfigParser()
    config.read('./config/' + config_file)
    configs = []
    for s, f in sections_fields:
        configs.append(config[s][f])
    return configs

def load_obj(name):
    """
    Load the pkl object by name
    :param name: name of file
    :return:
    """
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)