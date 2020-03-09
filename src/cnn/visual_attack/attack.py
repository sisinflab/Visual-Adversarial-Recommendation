from cleverhans.attacks import FastGradientMethod, ProjectedGradientDescent
from cleverhans.model import CallableModelWrapper
from cleverhans.utils_pytorch import convert_pytorch_model_to_tf
from torchvision import transforms
import tensorflow as tf
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'

class VisualAttack:
    def __init__(self, df_classes, num_classes, origin_class, target_class, model, params, attack_type):
        self.origin_class = origin_class
        self.target_class = target_class
        self.df_classes = df_classes
        self.params = params
        self.num_classes = num_classes
        self.attack_type = attack_type

        self.tf_model = convert_pytorch_model_to_tf(model)
        self.cleverhans_model = CallableModelWrapper(self.tf_model, output_layer='logits')
        self.sess = tf.compat.v1.Session()
        self.x_op = tf.placeholder(tf.float32, shape=(1, 3, None, None))

        if self.attack_type == 'fgsm':
            self.attack_op = FastGradientMethod(self.cleverhans_model, sess=self.sess)
            self.adv_x_op = self.attack_op.generate(self.x_op, **self.params)

        self.y_target = np.zeros((1, 1000))
        self.one_hot_encoded()
        self.params["y_target"] = self.y_target

    def must_attack(self, filename):
        print(self.df_classes.loc[self.df_classes["ImageID"] == filename]["ClassNum"].values())
        if int(self.df_classes.loc[self.df_classes["ImageID"] == filename]["ClassNum"].values()) == self.origin_class:
            return True
        else:
            return False

    def one_hot_encoded(self):
        self.y_target[0, self.target_class] = 1

    def run_fgsm(self, image):
        adv_img = self.sess.run(self.adv_x_op, feed_dict={self.x_op: image[None, ...]})
        adv_img_out = transforms.ToTensor()(adv_img[0])
        adv_img_out = adv_img_out.permute(1, 2, 0)
        return adv_img_out

    def run_pgd(self, image):
        pass

    def run_c_w(self, image):
        pass

    def run_deep_fool(self, image):
        pass