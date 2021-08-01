from torchvision import transforms
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torch import nn
import torch.nn.functional as F

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
        self.train_image_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.ColorJitter(brightness=.1, contrast=.1, hue=.1),
                transforms.RandomResizedCrop(size=(195, 100), scale=(0.9, 1.1), ratio=(0.4, 0.6)),
                transforms.Grayscale(),
                transforms.Resize((28, 28)),
                transforms.Normalize((0.5,), (0.5,))
            ])
        self.val_image_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Grayscale(),
                transforms.Resize((28, 28)),
                transforms.Normalize((0.5,), (0.5,))
            ])

    def data_prep(self, dataset_path="../Assets", model_output_path="../models"):
        # prepare the data and the transforms
        self.pt_path = model_output_path
        self.bs = 150
        self.train_data = FolderDataset(dataset_path, transforms=self.train_image_transform, train=True, generate_number_of_images=50)
        self.val_data = FolderDataset(dataset_path, transforms=self.val_image_transform, train=False)

        self.class_num = self.train_data.get_class_num()
        self.train_data_size = len(self.train_data)
        self.valid_data_size = len(self.val_data)
        # Create iterators for the Data loaded using DataLoader module
        self.train_data_loader = DataLoader(self.train_data, batch_size=self.bs, shuffle=True)
        self.valid_data_loader = DataLoader(self.val_data, batch_size=self.bs, shuffle=False)

        dataset = iter(self.train_data_loader)
        images, labels = next(dataset)
        if self.tensorboard:
            fig = plt.figure(figsize=(15, 4))
            for idx in np.arange(16):
                ax = fig.add_subplot(2, 8, idx + 1, xticks=[], yticks=[])
                plt.imshow(im_convert(images[idx]))
                ax.set_title([labels[idx].item()])
            plt.savefig('image.png', dpi=90, bbox_inches='tight')
            plt.show()

    def model_prep(self, resnet_type=None, input_size=28*28, output_size=11):
        """
        Function to prepare the model
        :param resnet_type: type of resnet model
        """
        # Parameters
        size_0 = 125
        size_1 = 65
        self.epoch = 10

        # Define Model
        self.model = LinearNN(input_size, size_0, size_1, input_size)
        self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def training(self, save_type="None"):
        self.running_loss_history = []
        self.running_corrects_history = []
        self.val_running_loss_history = []
        self.val_running_corrects_history = []
        for e in range(self.epoch):
            # parameters
            running_loss = 0.0
            running_corrects = 0.0
            running_loss_val = 0.0
            running_corrects_val = 0.0

            # loops
            for i, data in enumerate(self.train_data_loader):
                loss, corrects = run_data_to_model(data, self.device, self.model, self.criterion, self.optimizer)
                running_loss += loss
                running_corrects += corrects
            for i, data in enumerate(self.valid_data_loader ):
                loss, corrects = run_data_to_model(data, self.device, self.model, self.criterion, self.optimizer, train=False)
                running_loss_val += loss
                running_corrects_val += corrects
            epoch_loss = running_loss/len(self.train_data_loader)
            epoch_acc = running_corrects.item()/len(self.train_data_loader)
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
            save_model(model=self.model, type=save_type)

