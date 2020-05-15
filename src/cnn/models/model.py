import torch
import numpy as np
import os
import collections

pre_trained_models = {
    'free_adv': {
        'name': 'state_dict',
        'multi_gpu': True
    },
    'madry': {
        'name': 'model',
        'multi_gpu': True
    }
}


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

    def __init__(self, model=None, do_eval=True, gpu=0, model_path=None, model_name='ResNet50', pretrained_name=None):
        """
        Args:
            model: pytorch-like model (e.g. resnet50)
            do_eval (bool): True to use the model in inference-mode, otherwise False
            gpu (int): index of gpu to use (-1 for cpu)
            model_path (str): model path (when it is loaded from memory, e.g. your custom trained model
        """
        self.eval = do_eval
        self.gpu = gpu
        self.model_path = model_path
        self.feature_model = None

        if self.gpu != -1:
            use_cuda = torch.cuda.is_available()
            self.device = torch.device("cuda:" + str(self.gpu) if use_cuda else "cpu")
        else:
            self.device = torch.device("cpu")

        # if model is loaded from path
        if self.model_path:
            try:
                pretrained_model = torch.load(self.model_path, map_location=self.device).get(
                    pre_trained_models[pretrained_name]['name'])

                # if the model has been pretrained on multi gpu
                if pre_trained_models[pretrained_name]['multi_gpu']:
                    keys = pretrained_model.keys()
                    values = pretrained_model.values()

                    new_state_keys = []
                    new_values = []
                    for key, value in zip(keys, values):

                        # this is to make madrylab pre-trained model conformable to ResNet50 architecture from PyTorch
                        # as a matter of facts, in our configuration, we don't need any "attacker" and/or pre-processing
                        # "normalizer", we just want to load the pre-trained weights for the defense
                        if pretrained_name == 'madry':
                            if 'normalizer' not in key and 'attacker' not in key:
                                new_state_keys.append(key[len('module.model.'):])
                                new_values.append(value)

                        else:
                            new_state_keys.append(key[len('module.'):])
                            new_values = values

                    new_dict = collections.OrderedDict(list(zip(new_state_keys, new_values)))
                    model.load_state_dict(new_dict)

                # otherwise, the model has not been pretrained on multi gpu
                else:
                    model.load_state_dict(pretrained_model)

                self.model = model
                self.model.to(self.device)
                print('Loaded model from %s on %s' % (self.model_path, self.device))

            except FileNotFoundError:
                raise FileNotFoundError('Model file not found!')

        # otherwise, use pytorch pre-trained model
        else:
            self.model = model
            self.model.to(self.device)
            self.model_name = model_name
            print('Loaded PyTorch pre-trained %s on %s' % (self.model_name, self.device))

        #####################################################################################
        # OLD MODEL LOADING WITHOUT THE POSSIBILITY OF LOADING FROM MEMORY
        # if self.gpu != -1:
        #     use_cuda = torch.cuda.is_available()
        #     self.device = torch.device("cuda:" + str(self.gpu) if use_cuda else "cpu")
        #     self.model.to(self.device)
        # else:
        #     self.device = torch.device("cpu")
        #     self.model.to(self.device)

        # print(self.device)
        #####################################################################################

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

