import os
import time
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from nets.coco_dataset import COCODataset
from nets.sesf_net import SESFuseNet
from nets.nets_utility import *
import torch.nn as nn
from nets.lp_lssim_loss import LpLssimLoss

# parameter for net
experiment_name = 'lp+lssim_se_sf_net_times30'
gpu_device = "cuda:0"
# gpu_device_for_parallel = [2, 3]
learning_rate = 1e-4
epochs = 30
batch_size = 48
display_step = 100
shuffle = True
attention = 'cse'
# address
project_addrsss = os.getcwd()
train_dir = os.path.join(project_addrsss, "data", "coco2014", "train2014")
val_dir = os.path.join(project_addrsss, "data", "coco2014", "val2014")
log_address = os.path.join(project_addrsss, "nets", "train_record", experiment_name + "_log_file.txt")
is_out_log_file = True
parameter_address = os.path.join(project_addrsss, "nets", "parameters")

# datasets
data_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4500517361627943], [0.26465333914691797]),
])

image_datasets = {}
image_datasets['train'] = COCODataset(train_dir, transform=data_transforms, need_crop=False, need_augment=False)
image_datasets['val'] = COCODataset(val_dir, transform=data_transforms, need_crop=False, need_augment=False)

dataloders = {}
dataloders['train'] = DataLoader(
    image_datasets['train'],
    batch_size=batch_size,
    shuffle=shuffle,
    num_workers=1)
dataloders['val'] = DataLoader(
    image_datasets['val'],
    batch_size=batch_size,
    shuffle=shuffle,
    num_workers=1)
datasets_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
print_and_log("datasets size: {}".format(datasets_sizes), is_out_log_file, log_address)

# models
training_setup_seed(1)  # setup seed for all parameters, numpy.random, random, pytorch
# model = UNet(in_channel=1, out_channel=1)
model = SESFuseNet(attention)
model.to(gpu_device)
# model = nn.DataParallel(model, device_ids=gpu_device_for_parallel)

criterion = LpLssimLoss().to(gpu_device)
optimizer = optim.Adam(model.parameters(), learning_rate)
# optimizer = nn.DataParallel(optimizer, device_ids=gpu_device_for_parallel)


def val():
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for i, data in enumerate(dataloders['val']):
            input = data.to(gpu_device)
            optimizer.zero_grad()
            output = model.forward('train', input)
            loss, lp_loss, lssim_loss = criterion(image_in=input, image_out=output)
            running_loss += loss.item()

    epoch_loss_val = running_loss / datasets_sizes['val']
    return epoch_loss_val


def train(epoch):
    iterations_loss_list = []
    iterations_lp_loss_list = []
    iterations_lssim_loss_list = []
    model.train()
    adjust_learning_rate(optimizer, learning_rate, epoch)
    print_and_log('Train Epoch {}/{}:'.format(epoch + 1, epochs), is_out_log_file, log_address)
    running_loss = 0.0

    # Iterate over data.
    for i, data in enumerate(dataloders['train']):
        input = data.to(gpu_device)
        output = model.forward('train', input)
        loss, lp_loss, lssim_loss = criterion(image_in=input, image_out=output)
        running_loss += loss.item()

        if i % display_step == 0:
            print_and_log('\t{} {}-{}: Loss: {:.4f}'.format('train', epoch + 1, i, loss.item() / batch_size),
                          is_out_log_file, log_address)
            iterations_loss_list.append(loss.item() / batch_size)
            iterations_lp_loss_list.append(lp_loss.item() / batch_size)
            iterations_lssim_loss_list.append(lssim_loss.item() / batch_size)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    epoch_loss_train = running_loss / datasets_sizes['train']
    plot_iteration_loss(experiment_name, epoch + 1, iterations_loss_list, iterations_lp_loss_list,
                        iterations_lssim_loss_list)
    return epoch_loss_train


def main():
    min_loss = 100000000.0
    loss_train = []
    loss_val = []
    since = time.time()
    for epoch in range(epochs):
        epoch_loss_train = train(epoch)
        loss_train.append(epoch_loss_train)
        epoch_loss_val = val()
        loss_val.append(epoch_loss_val)
        print_and_log('\ttrain Loss: {:.6f}'.format(epoch_loss_train), is_out_log_file, log_address)
        print_and_log('\tvalidation Loss: {:.6f}'.format(epoch_loss_val), is_out_log_file, log_address)

        # deep copy the models
        if epoch_loss_val < min_loss:
            min_loss = epoch_loss_val
            best_model_wts = model.state_dict()
            print_and_log("Updating", is_out_log_file, log_address)
            torch.save(best_model_wts,
                       os.path.join(parameter_address, experiment_name + '.pkl'))
        plot_loss(experiment_name, epoch, loss_train, loss_val)
        # save models
        model_wts = model.state_dict()
        torch.save(model_wts,
                   os.path.join(parameter_address, experiment_name + "_" + str(epoch) + '.pkl'))

        time_elapsed = time.time() - since
        print_and_log('Time passed {:.0f}h {:.0f}m {:.0f}s'.
                      format(time_elapsed // 3600, (time_elapsed % 3600) // 60, time_elapsed % 60), is_out_log_file,
                      log_address)
        print_and_log('-' * 20, is_out_log_file, log_address)
    print_and_log("train loss: {}".format(loss_train), is_out_log_file, log_address)
    print_and_log("val loss: {}".format(loss_val), is_out_log_file, log_address)
    print_and_log("min val loss: {}".format(min(loss_val)), is_out_log_file, log_address)


if __name__ == "__main__":
    main()
