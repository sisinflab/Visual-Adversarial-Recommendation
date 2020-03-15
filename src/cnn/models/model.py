import torch
import numpy as np
import os


class Model:
    """
    This class represents a model. You can load a pretrained model (e.g. resnet50) or from the memory
    (by specifying its path) and use it for training/testing purposes.
    Attributes:
        model: pytorch-like model (e.g. resnet50)
        do_eval: True to use the model in inference-mode, otherwise False
        gpu (int): index of gpu to use (-1 for cpu)
        model_path (str): model path (when it is loaded from memory, e.g. your custom trained model
    """

    def __init__(self, model=None, do_eval=True, gpu=0, model_path=None):
        """
        Args:
            model: pytorch-like model (e.g. resnet50)
            do_eval (bool): True to use the model in inference-mode, otherwise False
            gpu (int): index of gpu to use (-1 for cpu)
            model_path (str): model path (when it is loaded from memory, e.g. your custom trained model
        """
        self.model = model
        self.eval = do_eval
        self.gpu = gpu
        self.model_path = model_path
        self.feature_model = None

        if self.gpu != -1:
            use_cuda = torch.cuda.is_available()
            self.device = torch.device("cuda:" + str(self.gpu) if use_cuda else "cpu")
            self.model.to(self.device)
        else:
            self.device = torch.device("cpu")
            self.model.to(self.device)

        print(self.device)

        if self.eval:
            self.model.eval()

    def set_out_layer(self, drop_layers):
        """
        Args:
            drop_layers (int): index of layers to drop from model (inverse order)
        """
        self.feature_model = torch.nn.Sequential(*list(self.model.children())[:-drop_layers])

        if self.eval:
            self.feature_model.eval()

    def classification(self, list_classes, sample):
        """
        This function runs classification given a model, the list of possible classes (as strings)
        and the input.

        Args:
            list_classes (list): list of possible classes (as strings)
            sample: tuple (sample, sample_filename)
        Return:
            A dictionary with ImageID, class (as string), class (as number), probability of classification.
        """
        image, filename = sample
        output = torch.nn.functional.softmax(input=self.model(image[None, ...].to(self.device)), dim=1)

        if self.model.training:
            print("Run model in inference mode!")
            exit(0)

        return {'ImageID': os.path.splitext(filename)[0],
                'ClassStr': list_classes[int(np.argmax(output.data.cpu().numpy()))],
                'ClassNum': np.argmax(output.data.cpu().numpy()),
                'Prob': np.amax(output.data.cpu().numpy())}

    def feature_extraction(self, sample):
        """
        This function runs feature extraction given a model and the input sample.
        Args:
            sample: tuple (sample, sample_filename)
        Return:
           The extracted feature.
        """
        image, filename = sample

        if self.feature_model.training:
            print("Run feature model in inference mode!")
            exit(0)

        if self.feature_model:
            feature = np.squeeze(self.feature_model(image[None, ...].to(self.device)).data.cpu().numpy())
            return feature
