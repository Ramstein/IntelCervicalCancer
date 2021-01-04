import multiprocessing
import os
from os.path import join

import torch
import torch.nn as nn
import torch.utils.data as data_utils
from torch.autograd.variable import Variable

from src.scripts.bbox_data_preproc import load_dataset
from src.scripts.models import Detector
from src.scripts.utils import model_save_dir, data_dir, batch_size, n_epoch, img_size, LR


def get_model():
    model = Detector()

    optimizer = torch.optim.Adam(model.parameters(), LR)
    model = model.cuda()
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model, device_ids=[id for id in range(torch.cuda.device_count())])
    criterion = nn.L1Loss().cuda()
    return model, criterion, optimizer


def train(model, train_loader):
    model.train()
    train_acc, correct_train, train_loss, target_count = 0, 0, 0, 0
    for i, (input, target) in enumerate(train_loader):
        target = target.cuda()
        input_var = Variable(input)
        target_var = Variable(target)

        optimizer.zero_grad()
        output = model(input_var)
        loss = criterion(output, target_var)
        loss.backward()
        optimizer.step()

        # accuracy
        _, predicted = torch.max(output.data, 1)
        target_count += target_var.size(0)
        correct_train += (target_var == predicted).sum().item()
        train_acc = (100 * correct_train) / target_count
    return train_acc, train_loss / target_count


def validate(model, val_loader):
    model.eval()
    val_acc, correct_val, val_loss, target_count = 0, 0, 0, 0
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda()
        input_var = Variable(input, volatile=True)
        target_var = Variable(target, volatile=True)
        output = model(input_var)
        loss = criterion(output, target_var)
        val_loss += loss.item()

        # accuracy
        _, predicted = torch.max(output.data, 1)
        target_count += target_var.size(0)
        correct_val += (target_var == predicted).sum().item()
        val_acc = 100 * correct_val / target_count
    return (val_acc * 100) / target_count, val_loss / target_count


if __name__ == "__main__":

    workers = multiprocessing.cpu_count()

    if not os.path.isdir(model_save_dir):
        os.makedirs(model_save_dir)

    """LOAD DATA"""
    """LOAD DATA"""
    X_train, y_train, X_val, y_val = load_dataset(data_dir, img_size, N_max=15616)  # total 15664 here 15616/64=244

    train_tensor = data_utils.TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    train_loader = data_utils.DataLoader(train_tensor, batch_size=batch_size, shuffle=True, pin_memory=True,
                                         num_workers=workers)

    val_tensor = data_utils.TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    val_loader = data_utils.DataLoader(val_tensor, batch_size=batch_size, shuffle=False, pin_memory=True,
                                       num_workers=workers)

    """MODEL"""
    """MODEL"""

    model, criterion, optimizer = get_model()
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    """TRAIN"""
    """TRAIN"""
    train_acc_old = 0
    epoch_str = ""
    for epoch in range(1, n_epoch + 1):
        train_acc, train_loss = train(model, train_loader)
        val_acc, val_loss = validate(model, val_loader)
        epoch_str = "epoch-{}_TA-{:.4f}_TL-{:.4f}_VA-{:.4f}_VL-{:.4f}".format(epoch, train_acc, train_loss, val_acc,
                                                                              val_loss)
        print(epoch_str)
        if train_acc_old < train_acc:
            torch.save(model.state_dict(), join(model_save_dir, 'model_TA-' + epoch_str + '.pth.tar'))
            print("Model " + 'model_TA-' + epoch_str + '.pth.tar' + " saved.")
        train_acc_old = train_acc

    torch.save(model.state_dict(), join(model_save_dir, 'model_TA-' + epoch_str + '.pth.tar'))
    print("Model " + 'model_TA-' + epoch_str + '.pth.tar' + " saved.")

    val_acc, val_loss = validate(model, val_loader)
    print("val_acc: ", val_acc, ", val_loss: ", val_loss)

    # """SAVE MODEL"""
    # """SAVE MODEL"""
    # state = {
    #     'epoch': epoch + 1,
    #     'arch': train_name,
    #     'state_dict': model.state_dict(),
    #     'val_loss': val_loss,
    #     'img_size': img_size
    # }
    # torch.save(state, join(model_save_dir, 'model.pth.tar'))
    #
    # """LOAD MODEL"""
    # """LOAD MODEL"""
    # state = torch.load(join(model_save_dir, 'model.pth.tar'))
    # print(state['arch'], state['val_loss'], state['epoch'])
    # model.load_state_dict(state['state_dict'])
    #
    # model.eval()
    # for i, (input, target) in enumerate(val_loader):
    #     target = target.cuda()
    #     input_var = Variable(input, volatile=True)
    #     target_var = Variable(target, volatile=True)
    #     output = model(input_var)
    #     break
    #
    # for img_num in range(16):
    #     netout = output.data.cpu().numpy()[img_num]
    #     points = netout2points(netout)
    #     img = input_var.data.cpu().numpy()[img_num]
    #     img = img.transpose((1, 2, 0)).astype(np.uint8)
    #     show_bgr(check_poly(img, points))
