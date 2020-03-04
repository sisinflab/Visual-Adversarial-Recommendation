from cnn.models.dataset import *
from cnn.models.model import *
from utils.read import *
from utils.write import *
import torchvision.models as models
import pandas as pd
import sys

def classify_and_extract():
    path_images, path_output_classes, path_classes = read_config(config_file='./config/configs.ini',
                                                                 sections_fields=[('PATHS', 'InputAmazonMenImages'),
                                                                                  ('PATHS', 'OutputAmazonMenClasses'),
                                                                                  ('PATHS', 'ImagenetClasses')])
    data = CustomDataset(root_dir=path_images)
    model = Model(model=models.resnet50(pretrained=True))
    img_classes = read_imagenet_classes_txt(path_classes)

    df = pd.DataFrame([], columns={'ImageID', 'ClassStr', 'ClassNum'})

    for i, d in enumerate(data):
        out_class = model.classification(list_classes=img_classes, sample=d)
        feature = model.feature_extraction(sample=d)
        df = df.append(out_class, ignore_index=True)
        sys.stdout.write('\r%d/%d samples completed' % (i + 1, data.num_samples))
        sys.stdout.flush()

    df = df.sort_values(by=['ImageID'])
    write_csv(df, path_output_classes)

if __name__ == '__main__':
    classify_and_extract()