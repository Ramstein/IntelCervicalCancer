import os
import pickle
from os.path import join

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.scripts.bbox_data_preproc import square_image, netout2points, check_poly
from src.scripts.kfold_data import load_train_add_kfold_df, get_test_df
from src.scripts.models import Detector
from src.scripts.utils import show_bgr, mkdir, CLASSES
from src.scripts.utils import SageMakerRoot_dir


def get_model_from_state(model_dir):
    from src.scripts.models import Detector
    model = Detector()
    model = torch.nn.DataParallel(model).cuda()
    state = torch.load(join(model_dir, 'model.pth.tar'))
    model.load_state_dict(state['state_dict'])
    model.eval()
    return model


def model_pred_points(img_path, model, detect_size):
    orig_img = cv2.imread(img_path)
    img = square_image(orig_img)
    square_size = img.shape[0]
    img = cv2.resize(img, (detect_size, detect_size))
    img = img.transpose((2, 0, 1))
    img = img.astype(np.float32)

    net_input = torch.from_numpy(img[np.newaxis])
    net_input = torch.autograd.Variable(net_input)

    output = model(net_input)
    output = output.cpu().data.numpy()[0]
    output *= square_size / detect_size

    y_shift = (square_size - orig_img.shape[0]) // 2
    x_shift = (square_size - orig_img.shape[1]) // 2
    output[0] -= x_shift
    output[1] -= y_shift

    return netout2points(output)


def pred_bboxes(df, model_dir):
    df_result = df.copy()
    model = Detector()
    model = torch.nn.DataParallel(model).cuda()
    state = torch.load(join(model_dir, 'model.pth.tar'))
    model.load_state_dict(state['state_dict'])
    model.eval()
    detect_size = state['img_size'][0]
    for i, row in df.iterrows():
        points = model_pred_points(row.Path, model, detect_size)
        min_x, min_y = points[0].astype(int)
        max_x, max_y = points[2].astype(int)

        df_result.loc[i, 'min_x'] = min_x
        df_result.loc[i, 'min_y'] = min_y
        df_result.loc[i, 'bbox_width'] = max_x - min_x
        df_result.loc[i, 'bbox_height'] = max_y - min_y
    df_result.min_x = df_result.min_x.astype(int)
    df_result.min_y = df_result.min_y.astype(int)
    df_result.bbox_width = df_result.bbox_width.astype(int)
    df_result.bbox_height = df_result.bbox_height.astype(int)

    return df_result


def get_crop_img(row, shift_scale):
    img = cv2.imread(row.Path)
    shift_x = int(shift_scale * (row.bbox_width / 2))
    shift_y = int(shift_scale * (row.bbox_height / 2))
    min_x, min_y = row.min_x - shift_x, row.min_y - shift_y
    max_x = row.min_x + row.bbox_width + shift_x
    max_y = row.min_y + row.bbox_height + shift_y
    min_x, max_x = np.clip([min_x, max_x], 0, row.Width)
    min_y, max_y = np.clip([min_y, max_y], 0, row.Height)
    img = img[min_y:max_y, min_x:max_x]
    return img


def add_postfix_if_exists(file_path, postfix='_'):
    while os.path.isfile(file_path):
        file_path = postfix.join(os.path.splitext(file_path))
    return file_path


def crop_save_data_df(df, save_dir, shift_scale, size):
    mkdir(save_dir)
    result_lst = []
    for i, row in df.iterrows():
        img = get_crop_img(row, shift_scale)
        img = cv2.resize(img, size[::-1])
        img_path = join(save_dir, row.Name)
        img_path = add_postfix_if_exists(img_path)
        cv2.imwrite(img_path, img)

        result_row = row.copy()
        result_row.Path = img_path
        result_lst.append(result_row)
    return pd.DataFrame(result_lst)


def save_crop_dataset(folder_dict, save_dir, shift_scale=1.0, size=(256, 256)):
    result_folder_dict = dict()
    for folder, df_lst in folder_dict.items():
        folder_path = join(save_dir, folder)
        folder_df_lst = []
        for df in df_lst:
            if 'Class' in df.columns:
                for cls in CLASSES:
                    cls_path = join(folder_path, cls)
                    cls_df = df[df.Class == cls]
                    result_df = crop_save_data_df(cls_df, cls_path, shift_scale, size)
                    folder_df_lst.append(result_df)
            else:
                result_df = crop_save_data_df(df, folder_path, shift_scale, size)
                folder_df_lst.append(result_df)

        result_folder_dict[folder] = pd.concat(folder_df_lst)

    with open(join(save_dir, 'data_df_dict.pickle'), 'wb') as f:
        pickle.dump(result_folder_dict, f)

    return result_folder_dict


if __name__ == "__main__":
    dataset_name = 'data001_size224'
    train_name = 'detector_002'
    model_dir = os.path.join(SageMakerRoot_dir, 'models/{0}/{1}/'.format(dataset_name, train_name))

    train_df, add_df = load_train_add_kfold_df()
    test_df = get_test_df()

    test_pred_df = pred_bboxes(test_df, model_dir)
    add_pred_df = pred_bboxes(add_df, model_dir)

    folder_dict = {
        'train': 4 * [train_df] + [add_pred_df],
        'test': [test_pred_df]
    }

    """first sample plot"""
    size = (1024, 1024)

    save_dir = os.path.join(SageMakerRoot_dir, 'preproc_data/data002_kfold_val_detector002_size256_scale1.5')

    result_folder_dict = save_crop_dataset(folder_dict, save_dir, shift_scale=1.5, size=size)

    folder_dict = {
        'train': 4 * [train_df] + [add_pred_df],
        'test': [test_pred_df]
    }

    """second sample plot"""
    save_dir = os.path.join(SageMakerRoot_dir, 'preproc_data/data002_kfold_val_detector002_size256_scale1.0')

    result_folder_dict = save_crop_dataset(folder_dict, save_dir, shift_scale=1.0, size=size)

    folder_dict = {
        'train': 4 * [train_df] + [add_pred_df],
        'test': [test_pred_df]
    }

    """third sample plot"""
    save_dir = os.path.join(SageMakerRoot_dir, 'preproc_data/data002_kfold_val_detector002_size256_scale2.0')

    result_folder_dict = save_crop_dataset(folder_dict, save_dir, shift_scale=2.0, size=size)

    model = get_model_from_state(model_dir)
    img_path = os.path.join(SageMakerRoot_dir, 'data/test/348.jpg')

    for img_path in test_df.Path:
        points = model_pred_points(img_path, model, 224)
        orig_img = cv2.imread(img_path)
        show_bgr(check_poly(orig_img, points))
        plt.show()
