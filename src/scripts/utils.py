import os
from os.path import join

import matplotlib.pyplot as plt

# SageMakerTrainingRoot_dir = "/opt/ml/code/"  # Here /code/==/IntelCervicalCancer/
SageMakerTrainingRoot_dir = ""

if SageMakerTrainingRoot_dir:
    SageMakerRoot_dir = SageMakerTrainingRoot_dir
else:
    SageMakerRoot_dir = "/home/ec2-user/SageMaker/IntelCervicalCancer"

TRAIN_DATA_DIR = join(SageMakerRoot_dir, 'data/train/')
TEST_DATA_DIR = join(SageMakerRoot_dir, 'data/test/')
ADD_DATA_DIR = join(SageMakerRoot_dir, 'data/additional/')
BBOX_FILES = join(SageMakerRoot_dir, 'data/bboxes/%s_bbox.tsv')
SAMPLE_PATH = join(SageMakerRoot_dir, 'data/sample_submission.csv')
dataset_in_pickle = join(SageMakerRoot_dir, 'data/train_val.pickle')
aug_output_dir = join(SageMakerRoot_dir, 'data/detect_data/data001_size1024')

CLASSES = ['Type_1', 'Type_2', 'Type_3']


def mkdir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def show_bgr(img):
    plt.figure(figsize=(7, 7))
    plt.imshow(img[:, :, (2, 1, 0)])
