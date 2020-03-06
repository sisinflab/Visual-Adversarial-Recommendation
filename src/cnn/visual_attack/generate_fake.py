import torch
import torchvision.models as models
import numpy as np
import pandas as pd
import os
import tensorflow as tf
import sys
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from cleverhans.attacks import FastGradientMethod, ProjectedGradientDescent
from cleverhans.model import CallableModelWrapper
from cleverhans.utils import AccuracyReport
from cleverhans.utils_pytorch import convert_pytorch_model_to_tf
from matplotlib import colors, cm, pyplot as plt
from torch.utils import data
from PIL import Image
import torchvision

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'


class MyImageFolder(ImageFolder):
    def __init__(self, **kwargs):
        super(MyImageFolder, self).__init__(**kwargs)

    def __getitem__(self, index):
        try:
            return super(MyImageFolder, self).__getitem__(index)
        except IOError:
            with open('/home/felice/Projects/VisualRSAdvAttacks/cnn/damaged_images.txt', 'a') as damaged_images_file:
                path, _ = self.imgs[index]
                damaged_images_file.write(str(os.path.basename(path)) + '\n')
            return super(MyImageFolder, self).__getitem__(index + 1)


def load_dataset(data_path):
    train_dataset = MyImageFolder(root=data_path,
                                  transform=transforms.Compose([  # transforms.Resize(256),
                                      # transforms.CenterCrop(224),
                                      transforms.ToTensor()])
                                  )

    # , transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]
    train_loader = DataLoader(dataset=train_dataset,
                              pin_memory=True)
    return train_loader


infer_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.ToTensor()
])

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

resnet50 = models.resnet50(pretrained=True)
resnet50.to(device)
resnet50.eval()

images = load_dataset('./data/amazon_men/images_MEN_category_under_attack')
images_paths = os.listdir('./data/amazon_men/images_MEN_category_under_attack/img')
images_paths.sort()
classes_txt = './data/amazon_men/imagenet1000_clsidx_to_labels.txt'

sess = tf.compat.v1.Session()
x_op = tf.placeholder(tf.float32, shape=(1, 3, None, None))

# Convert pytorch model to a tf_model and wrap it in cleverhans
tf_model_fn = convert_pytorch_model_to_tf(resnet50)
cleverhans_model = CallableModelWrapper(tf_model_fn, output_layer='logits')

fgsm_op = FastGradientMethod(cleverhans_model, sess=sess)
# pgd_op = ProjectedGradientDescent(cleverhans_model, sess=sess)
y_target = np.zeros((1, 1000))
y_target[0, 770] = 1

fgsm_params = {'eps': 0.015686275,
               'y_target': y_target,
               'clip_min': 0,
               'clip_max': 1
               }

pgd_params = {'eps': 0.015686275,
              'y_target': y_target,
              'clip_min': 0,
              'clip_max': 1,
              'nb_iter': 5
              }

adv_x_op = fgsm_op.generate(x_op, **fgsm_params)
# pgd_adv_x_op = pgd_op.generate(x_op, **pgd_params)
adv_preds_op = tf_model_fn(adv_x_op)

total = 0
correct = 0

features_dir = './data/amazon_men/fgsm_features/'
features_model = torch.nn.Sequential(*list(resnet50.children())[:-1])
features_model.to(device)
features_model.eval()

resize = torchvision.transforms.Resize((256, 256))
crop = torchvision.transforms.CenterCrop(224)
to_tensor = torchvision.transforms.ToTensor()
normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

correctly_altered = 0
remain_correct = 0

for i, (xs, filename) in enumerate(zip(images, images_paths)):
    try:
        # Original should be 502
        img = Image.open('./data/amazon_men/images_MEN_category_under_attack/img/{0}'.format(filename))
        # resized = resize(img)
        # cropped = crop(img)
        to_tensored = to_tensor(img)
        # normalized = normalize(to_tensored)
        output = torch.nn.functional.softmax(input=resnet50(to_tensored[None, ...].to(device)), dim=1)
        original = np.argmax(output.data.cpu().numpy())
        # print(original)

        img_array = np.asarray(img)

        adv_img = sess.run(adv_x_op, feed_dict={x_op: to_tensored[None, ...]})
        # denormalized_perturbed = np.array([[0.229, 0.224, 0.225]]).reshape((1, 3, 1, 1)) * xs[0].cpu().numpy() + np.array([[0.485, 0.456, 0.406]]).reshape((1, 3, 1, 1)) + perturbation
        a = to_tensor(adv_img[0])
        a = a.permute(1, 2, 0)

        torchvision.utils.save_image(a, 'test/fgsm_{0}'.format(filename))
        torchvision.utils.save_image(to_tensored, 'test/{0}'.format(filename))

        # Image.fromarray(a.cpu().numpy()[0] * 255).save(fp='fgsm_{0}'.format(filename))
        # Image.fromarray(a.cpu().numpy()[0] * 255).show()

        # resized = resize(Image.fromarray(adv_img[0].cpu().numpy(), mode='RGB'))
        # cropped = crop(resized)
        # to_tensored = to_tensor(cropped)
        normalized = normalize(a)

        # adv_img = data.DataLoader(adv_img)


        output = torch.nn.functional.softmax(input=resnet50(normalized.unsqueeze(0).to(device)), dim=1)
        adversarial = np.argmax(output.data.cpu().numpy())

        if original == 502 and adversarial==770:
            output_features = features_model(normalized.unsqueeze(0).to(device))
            np.save(features_dir + os.path.splitext(filename)[0], output_features.data.cpu().numpy().reshape((1, 2048)))

        print("filename {0} - Original: {1} - Adversarial {2}".format(filename, original, adversarial))
        if adversarial == 770:
            correctly_altered += 1
        elif adversarial == 502:
            remain_correct += 1

        if i == 20:
            break

        # plt.show()
        # break
    except RuntimeError as err:
        print("Error on img {0}".format(filename))
print('Attack Precision: {0}\nStill Original: {1}'.format(correctly_altered / len(images_paths),
                                                          remain_correct / len(images_paths)))
