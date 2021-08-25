from torchvision import transforms
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torch import nn
import torch.nn.functional as F
from networks import pt_linear

import sys
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from data_loading.data_loaders import FolderDataset
from utils.primary import im_convert, run_data_to_model, save_model
from utils.visualisation import print_metrix, plot_metrix
from networks.custom_networks import LinearNN


class train():
    def __init__(self, tensorboard=False):
        self.tensorboard = tensorboard
        if tensorboard:
            self.writer = SummaryWriter("runs/nums")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def prep(self, dataset_path="../Assets", model_output_path="../models"):
        # prepare the data and the transforms
        self.pt_path = model_output_path
        self.bs = 50

        self.model = pt_linear.PtLinearNN(LinearNN, self.device, model_name="my_model", input_size=28)

        self.train_data = FolderDataset(dataset_path, transforms=self.model.get_train_transforms(), train=True, generate_number_of_images=100)
        self.val_data = FolderDataset(dataset_path, transforms=self.model.get_valid_transforms(), train=False)

        self.class_num = self.train_data.get_class_num()

        self.model.set_output_size(self.class_num)
        self.model.model_prep()

        self.train_data_size = len(self.train_data)
        self.valid_data_size = len(self.val_data)
        # Create iterators for the Data loaded using DataLoader module
        self.train_data_loader = DataLoader(self.train_data, batch_size=self.bs, shuffle=True)
        self.valid_data_loader = DataLoader(self.val_data, 1, shuffle=False)

        if self.tensorboard:
            dataset = iter(self.train_data_loader)
            images, labels = next(dataset)
            fig = plt.figure(figsize=(15, 4))
            for idx in np.arange(16):
                ax = fig.add_subplot(2, 8, idx + 1, xticks=[], yticks=[])
                plt.imshow(im_convert(images[idx]))
                ax.set_title([labels[idx].item()])
            plt.savefig('image.png', dpi=90, bbox_inches='tight')
            plt.show()

    def train(self, save_type="None", enabled_training=True):
        self.running_loss_history = []
        self.running_corrects_history = []
        self.val_running_loss_history = []
        self.val_running_corrects_history = []
        for e in range(self.model.epoch):
            # parameters
            running_loss = 0.0
            running_corrects = 0.0
            running_loss_val = 0.0
            running_corrects_val = 0.0

            # loops
            for i, data in enumerate(self.train_data_loader):
                loss, corrects = run_data_to_model(data, self.device, self.model.model, self.model.get_criterion(), self.model.get_optimizer(), train=enabled_training)
                running_loss += loss
                running_corrects += corrects
            for i, data in enumerate(self.valid_data_loader):
                loss, corrects = run_data_to_model(data, self.device, self.model.model, self.model.get_criterion(), self.model.get_optimizer(), train=False)
                running_loss_val += loss
                running_corrects_val += corrects
            epoch_loss = running_loss/len(self.train_data_loader)
            epoch_acc = running_corrects.item()/len(self.train_data_loader)/self.bs
            self.running_loss_history.append(epoch_loss)
            self.running_corrects_history.append(epoch_acc)

            val_epoch_loss = running_loss_val/len(self.valid_data_loader)
            val_epoch_acc = running_corrects_val.item()/len(self.valid_data_loader)
            self.val_running_loss_history.append(val_epoch_loss)
            self.val_running_corrects_history.append(val_epoch_acc)
            print_metrix(e, epoch_loss, epoch_acc, val_epoch_loss, val_epoch_acc)


        plot_metrix(self.running_loss_history, self.val_running_loss_history, self.running_corrects_history, self.val_running_corrects_history)
        if save_type == "None":
            pass
        else:
            save_model(model=self.model, type=save_type, model_name=self.model.get_model_name())

