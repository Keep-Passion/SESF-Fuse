import os
import cv2
import numpy as np
from IPython.display import clear_output
import matplotlib.pyplot as plt
import torch
import random


def training_setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def adjust_learning_rate(optimizer, learning_rate, epoch):
    """Sets the learning rate to the initial LR decayed by half every 10 epochs until 1e-5"""
    lr = learning_rate * (0.8 ** (epoch // 2))
    #     if not lr < 1e-6:
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def plot_loss(experiment_name, epoch, train_loss_list, val_loss_list):
    clear_output(True)
    print('Epoch %s. train loss: %s. val loss: %s' % (epoch, train_loss_list[-1], val_loss_list[-1]))
    print('Best val loss: %s' % (min(val_loss_list)))
    print('Back up')
    print('train_loss_list: {}'.format(train_loss_list))
    print('val_loss_list: {}'.format(val_loss_list))
    plt.figure()
    plt.plot(train_loss_list, color="r", label="train loss")
    plt.plot(val_loss_list, color="b", label="val loss")
    plt.legend(loc="best")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss " + experiment_name, fontsize=16)
    figure_address = os.path.join(os.path.join(os.getcwd(), 'nets'), 'figures')
    plt.savefig(os.path.join(figure_address, experiment_name + '_loss.png'))
    plt.show()


def plot_iteration_loss(experiment_name, epoach, loss, lp_loss, lssim_loss):
    # clear_output(True)
    # print('Iteration %s. loss: %s.' % (iteration, loss))
    # print('Iteration %s. lp_loss: %s.' % (iteration, lp_loss))
    # print('Iteration %s. lssim_loss: %s.' % (iteration, lssim_loss))
    # print('loss: {}'.format(loss))
    # print('lp_loss: {}'.format(lp_loss))
    # print('lssim_loss: {}'.format(lssim_loss))
    plt.figure()
    plt.plot(loss, color="r", label="loss")
    plt.plot(lp_loss, color="g", label="lp_loss")
    plt.plot(lssim_loss, color="b", label="lssim_loss")
    plt.legend(loc="best")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title("Loss " + experiment_name, fontsize=16)
    figure_address = os.path.join(os.path.join(os.getcwd(), 'nets'), 'figures')
    plt.savefig(os.path.join(figure_address, experiment_name + '_' + str(epoach) + '_loss.png'))
    plt.show()


def print_and_log(content, is_out_log_file=True, file_address=None):
    print(content)
    if is_out_log_file:
        f = open(os.path.join(file_address), "a")
        f.write(content)
        f.write("\n")
        f.close()


def get_mean_value(input_dir):
    images_list = [os.path.join(input_dir, item) for item in sorted(os.listdir(input_dir))]
    count = 0
    pixel_sum = 0
    for index, sub_folder in enumerate(images_list):
        image_name = os.path.basename(sub_folder)
        last_image = cv2.imread(os.path.join(sub_folder, image_name + "_1.png"), 0) * 1.0 / 255
        next_image = cv2.imread(os.path.join(sub_folder, image_name + "_2.png"), 0) * 1.0 / 255
        pixel_sum = pixel_sum + np.sum(last_image) + np.sum(next_image)
        count = count + last_image.size + next_image.size
    return pixel_sum / count


def get_std_value(input_dir, mean):
    images_list = [os.path.join(input_dir, item) for item in sorted(os.listdir(input_dir))]
    count = 0
    pixel_sum = 0
    for index, sub_folder in enumerate(images_list):
        image_name = os.path.basename(sub_folder)
        last_image = np.power((cv2.imread(os.path.join(sub_folder, image_name + "_1.png"), 0) * 1.0 / 255) - mean, 2)
        next_image = np.power((cv2.imread(os.path.join(sub_folder, image_name + "_2.png"), 0) * 1.0 / 255) - mean, 2)
        pixel_sum = pixel_sum + np.sum(last_image) + np.sum(next_image)
        count = count + last_image.size + next_image.size
    return np.sqrt(pixel_sum / count)
