from cleverhans.future.torch.attacks import *
from cleverhans.model import CallableModelWrapper
from cleverhans.utils_pytorch import convert_pytorch_model_to_tf
from torchvision import transforms
import numpy as np
import torch
import os
import logging

from cnn.visual_attack.carlini_wagner_l2_std import CarliniWagnerL2Std
from cnn.visual_attack.zoo_l2 import ZOOL2
from cnn.visual_attack.spsa_no_clip import SPSANoClip
from cnn.visual_attack.saliency_map_method_memory import SaliencyMapMethodMemory

logging.disable(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf


class VisualAttack:
    def __init__(self,
                 tf_pytorch,
                 df_classes,
                 num_classes,
                 origin_class,
                 target_class,
                 model,
                 device,
                 params,
                 attack_type):

        self.origin_class = origin_class
        self.target_class = target_class
        self.df_classes = df_classes
        self.params = params
        self.num_classes = num_classes
        self.attack_type = attack_type
        self.device = device

        # CHOOSE BETWEEN TENSORFLOW AND PYTORCH IMPLEMENTATION
        if tf_pytorch == 'pytorch':
            # NEW PYTORCH IMPLEMENTATION
            self.model = model
            self.model.to(self.device)

        elif tf_pytorch == 'tf':
            self.tf_model = convert_pytorch_model_to_tf(model)
            self.cleverhans_model = CallableModelWrapper(self.tf_model, output_layer='logits')
            self.sess = tf.Session()
            self.adv_x_op = None

            self.y_target = np.zeros((1, 1000), dtype=np.uint8)
            self.one_hot_encoded()

            if self.attack_type == 'cw':
                self.params.pop('y_target')
            elif self.attack_type == 'spsa':
                self.y_target_tf = tf.placeholder(tf.int64)
            else:
                self.params["y_target"] = self.y_target

        else:
            raise NotImplementedError('Library not recognized')

        # SET AND INITIALIZE ATTACK TYPE
        if self.attack_type == 'fgsm':
            print("\nSetting fgsm attack")
            # self.attack_op = FastGradientMethod(self.cleverhans_model, sess=self.sess)
        elif self.attack_type == 'cw':
            print("\nSetting carlini & wagner attack")
            # self.attack_op = CarliniWagnerL2(self.cleverhans_model, sess=self.sess)
            self.attack_op = CarliniWagnerL2Std(self.cleverhans_model, sess=self.sess)
        elif self.attack_type == 'pgd':
            print("\nSetting pgd attack")
            # self.attack_op = MadryEtAl(self.cleverhans_model, sess=self.sess)
        elif self.attack_type == 'jsma':
            print("\nSetting jsma attack")
            self.attack_op = SaliencyMapMethodMemory(self.cleverhans_model, sess=self.sess)
        elif self.attack_type == 'zoo':
            print("\nSetting zoo attack")
            self.batch_size = params["batch_size"]
            print("Batch size set to: %d" % self.batch_size)
            # self.attack_op = ZOOL2(model=self.tf_model,
            #                        sess=self.sess)
        elif self.attack_type == 'spsa':
            print("\nSetting spsa attack")
            self.batch_size = params["batch_size"]
            print("Batch size set to: %d" % self.batch_size)
            self.attack_op = SPSANoClip(model=self.cleverhans_model, sess=self.sess)
        else:
            raise NotImplementedError('Not implemented attack.')

    def must_attack(self, filename):
        if self.df_classes.loc[self.df_classes["ImageID"] == int(os.path.splitext(filename)[0]), "ClassNum"].item() == self.origin_class:
            return True
        else:
            return False

    def one_hot_encoded(self):
        self.y_target[0, self.target_class] = 1

    def run_attack(self, image):
        # RUN ATTACK DEPENDING ON ATTACK TYPE
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

        elif self.attack_type == 'cw':
            self.x_op = tf.placeholder(tf.float32, shape=(1, 3, None, None))
            self.x_op = tf.reshape(self.x_op, shape=(1, 3, image.shape[2], image.shape[3]))
            self._y_P = tf.placeholder(tf.float32, shape=(1, 1000))

            self.adv_x_op = self.attack_op.generate(self.x_op, y_target=self._y_P, **self.params)

            adv_img = self.sess.run(self.adv_x_op, feed_dict={self.x_op: image, self._y_P: self.y_target})
            adv_img_out = torch.from_numpy(adv_img)
            return adv_img_out

        elif self.attack_type == 'jsma':
            self.x_op = tf.placeholder(tf.float32, shape=(1, 3, None, None))
            self.x_op = tf.reshape(self.x_op, shape=(1, 3, image.shape[2], image.shape[3]))
            self.params['y_target'] = tf.cast(tf.convert_to_tensor(self.y_target), tf.int64)
            self.adv_x_op = self.attack_op.generate(self.x_op, **self.params)
            adv_img = self.sess.run(self.adv_x_op, feed_dict={self.x_op: image})
            adv_img_out = torch.from_numpy(adv_img)
            return adv_img_out

        elif self.attack_type == 'zoo':
            attack_op = ZOOL2(self.sess,
                              self.cleverhans_model,
                              height=image.shape[2],
                              width=image.shape[3],
                              batch_size=self.batch_size)
            adv_img = attack_op.attack(image, self.y_target)
            del attack_op
            adv_img_out = torch.from_numpy(adv_img).permute(0, 3, 1, 2)
            return adv_img_out

        elif self.attack_type == 'spsa':
            self.x_op = tf.placeholder(tf.float32, shape=(1, 3, None, None))
            self.x_op = tf.reshape(self.x_op, shape=(1, 3, image.shape[2], image.shape[3]))
            self.adv_x_op = self.attack_op.generate(x=self.x_op,
                                                    y_target=self.y_target_tf,
                                                    eps=self.params['eps'],
                                                    nb_iter=self.params['nb_iter'],
                                                    spsa_samples=self.batch_size)
            adv_img = self.sess.run(self.adv_x_op, feed_dict={self.x_op: image, self.y_target_tf: self.target_class})
            adv_img_out = torch.from_numpy(adv_img)
            return adv_img_out

        else:
            raise NotImplementedError("Attack not implemented yet.")

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

