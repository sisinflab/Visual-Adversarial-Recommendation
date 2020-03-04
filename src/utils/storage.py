import pickle


def save_obj(obj, name):
    """
    Store the object in a pkl file
    :param obj: python object to be stored
    :param name: file name (Not insert .pkl)
    :return:
    """
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    """
    Load the pkl object by name
    :param name: name of file ù
    :return:
    """
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)