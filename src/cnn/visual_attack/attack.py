from cleverhans.attacks import FastGradientMethod, MadryEtAl, CarliniWagnerL2, SaliencyMapMethod
from cleverhans.future.torch.attacks import *
from cleverhans.model import CallableModelWrapper
from cleverhans.utils_pytorch import convert_pytorch_model_to_tf
from torchvision import transforms
import numpy as np
import torch
import os
import logging

logging.disable(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf


class VisualAttack:
    def __init__(self, df_classes, num_classes, origin_class, target_class, model, device, params, attack_type):
        self.origin_class = origin_class
        self.target_class = target_class
        self.df_classes = df_classes
        self.params = params
        self.num_classes = num_classes
        self.attack_type = attack_type
        self.device = device

        # NEW PYTORCH IMPLEMENTATION
        self.model = model
        self.model.to(self.device)

        # OLD TENSORFLOW TO PYTORCH IMPLEMENTATION
        # self.tf_model = convert_pytorch_model_to_tf(model)
        # self.cleverhans_model = CallableModelWrapper(self.tf_model, output_layer='logits')
        # self.sess = tf.compat.v1.Session()
        # self.x_op = tf.placeholder(tf.float32, shape=(1, 3, None, None))
        # self.adv_x_op = None

        # self.y_target = np.zeros((1, 1000), dtype=np.uint8)
        # self.one_hot_encoded()
        # self.params["y_target"] = self.y_target

        if self.attack_type == 'fgsm':
            print("Setting fgsm attack")
            # self.attack_op = FastGradientMethod(self.cleverhans_model, sess=self.sess)
        elif self.attack_type == 'cw':
            print("Setting carlini & wagner attack")
            # self.attack_op = CarliniWagnerL2(self.cleverhans_model, sess=self.sess)
        elif self.attack_type == 'pgd':
            print("Setting pgd attack")
            # self.attack_op = MadryEtAl(self.cleverhans_model, sess=self.sess)
        elif self.attack_type == 'jsma':
            print("Setting jsma attack")
            # self.attack_op = SaliencyMapMethod(self.cleverhans_model, sess=self.sess)

    def must_attack(self, filename):
        if self.df_classes.loc[self.df_classes["ImageID"] == int(os.path.splitext(filename)[0]), "ClassNum"].item() == self.origin_class:
            return True
        else:
            return False

    def one_hot_encoded(self):
        self.y_target[0, self.target_class] = 1

    # NEW VERSION WITH PYTORCH MODEL
    def run_attack(self, image):
        if self.attack_type == 'fgsm':
            return fast_gradient_method(model_fn=self.model,
                                        x=image.to(self.device),
                                        eps=self.params["eps"],
                                        norm=self.params["ord"],
                                        clip_min=self.params["clip_min"],
                                        clip_max=self.params["clip_max"],
                                        targeted=True,
                                        y=torch.from_numpy(np.array([self.target_class])).to(self.device))

        elif self.attack_type == 'pgd':
            return projected_gradient_descent(model_fn=self.model,
                                              x=image.to(self.device),
                                              eps=self.params["eps"],
                                              eps_iter=self.params["eps_iter"],
                                              nb_iter=self.params["nb_iter"],
                                              norm=self.params["ord"],
                                              clip_min=self.params["clip_min"],
                                              clip_max=self.params["clip_max"],
                                              targeted=True,
                                              y=torch.from_numpy(np.array([self.target_class])).to(self.device))
        else:
            print("Attack not implemented yet.")
            exit(0)

    # OLD VERSION
    def run_attack_old(self, image):
        if self.attack_type == 'cw':
            # Obtain Image Parameters
            image = image.cpu().numpy()
            img_row, img_col, nchannel = image.shape[1], image.shape[2], image.shape[0]
            nb_classes = 1

            adv_inputs = np.array(
                [[instance] for
                 instance in [image]], dtype=np.float32)

            one_hot = np.zeros((1, 1000))
            one_hot[0, 770] = 1

            adv_inputs = adv_inputs.reshape(
                (nb_classes, nchannel, img_row, img_col))
            adv_ys = np.array([one_hot],
                              dtype=np.float32).reshape((nb_classes, 1000))

            self.x_op = tf.placeholder(tf.float32, shape=(1, 3, None, None))
            self.x_op = tf.reshape(self.x_op, shape=(nb_classes, 3, image.shape[1], image.shape[2]))
            self.params["y_target"] = adv_ys
            self.adv_x_op = self.attack_op.generate(self.x_op, **self.params)

            adv_img = self.sess.run(self.adv_x_op, feed_dict={self.x_op: adv_inputs})
            adv_img_out = transforms.ToTensor()(adv_img[0])
            adv_img_out = adv_img_out.permute(1, 2, 0)
            return adv_img_out

        elif self.attack_type == 'jsma':
            self.x_op = tf.reshape(self.x_op, shape=(1, 3, image.shape[1], image.shape[2]))
            self.y_target = tf.cast(tf.convert_to_tensor(self.y_target), tf.int64)
            self.params["y_target"] = self.y_target

        self.adv_x_op = self.attack_op.generate(self.x_op, **self.params)

        adv_img = self.sess.run(self.adv_x_op, feed_dict={self.x_op: image[None, ...]})
        adv_img_out = transforms.ToTensor()(adv_img[0])
        adv_img_out = adv_img_out.permute(1, 2, 0)

        return adv_img_out
