"""
Created by Merra Felice Antonio 08/04/2020
This pre processing step has to be executed in a directory with the ratings.csv files of the used dataset and all the images download in a directory (e.g., WOMEN).
There are a set of control to manage not-available/broken images that we have found diring the experiments
"""

import time
import sendmail
import numpy as np
import pandas as pd
import torch
import torchvision
import torchvision.models as models
from PIL import Image


dataset = 'WOMEN'
np.random.seed(1234)

start = time.time()

ratings = pd.read_csv('ratings.csv', usecols=['user', 'item', 'rating', 'timestamp'])

from os import listdir
from os.path import isfile, join

image_not_available = np.load('image_not_available.npy')
tolta1 = np.load('B00007GDIR.npy')
tolta2 = np.load('B000FK5R6W.npy')

onlyfiles_pre = [f.split('.')[0] for f in listdir(dataset) if isfile(join(dataset, f))]
total_images = len(onlyfiles_pre)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:1" if use_cuda else "cpu")

resnet50 = models.resnet50(pretrained=True)
resnet50.to(device)
resnet50.eval()
to_tensor = torchvision.transforms.ToTensor()

onlyfiles = []
cnt = 0

print('Start image elaboration')
start = time.time()
for index, img_name in enumerate(onlyfiles_pre):

    # 500000003X.jpg -> Image not available
    # Extract the feature
    try:
        img = Image.open('./{0}/{}.jpg'.format(dataset, img_name))
        if img.mode != 'RGB':
            print(img_name + ' Cast to RGB')
            img = img.convert('RGB')

        to_tensored = to_tensor(img)

        feature_model = torch.nn.Sequential(*list(resnet50.children())[:-1])

        feature = np.squeeze(feature_model(to_tensored[None, ...].to(device)).data.cpu().numpy())

        # class_image = torch.nn.functional.softmax(input=resnet50(to_tensored[None, ...].to(device)), dim=1)

        if sum(feature - image_not_available) == 0:
            print('Image_not_available ' + img_name)
            cnt += 1
        elif sum(feature - tolta1[0]) == 0:
            print('Errore 1 ' + img_name)
            cnt += 1
        elif sum(feature - tolta2[0]) == 0:
            print('Errore 2 ' + img_name)
            cnt += 1
        else:
            onlyfiles.append(img_name)

        if (index+1) % 1000 == 0 and index != 0:
            print('***** Elaborate {0}/{1} in {2}*****'.format(index+1, total_images, time.time() - start))
            start = time.time()
    except Exception as ex:
        print('Exception on {0} - {1}'.format(img_name, ex))

print('***** Number Broken Images: {0}'.format(cnt))

ratings = ratings[ratings['item'].isin(list(onlyfiles))]
ratings.to_csv('filtered_ratings.txt', sep='\t', header=None, index=None)
