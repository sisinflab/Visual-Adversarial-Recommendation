import tensorflow as tf


class RecommenderModel(tf.keras.Model):
    """
    This class represents a recommender model.
    You can load a pretrained model by specifying its ckpt path
     and use it for training/testing purposes.

    Attributes:
        model:
        do_eval: True to use the model in inference-mode, otherwise False
        gpu (int): index of gpu to use (-1 for cpu)
        model_path (str): model path
    """

    def __init__(self, data, path_output_rec_result, path_output_rec_weight, rec):
        self.rec = rec
        self.data = data
        self.num_items = data.num_items
        self.num_users = data.num_users
        self.path_output_rec_result = path_output_rec_result
        self.path_output_rec_weight = path_output_rec_weight

    def train(self):
        pass

    def restore(self):
        pass

