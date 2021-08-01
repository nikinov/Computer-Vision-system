from torchvision import transforms
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision

import sys
from PIL import Image
import matplotlib.pyplot as plt

from DataLoaders import FolderDataset
from networks import LinearNN


class train():
    def __init__(self, tensorboard=False):
        if tensorboard:
            self.writer = SummaryWriter("runs/nums")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_image_transform = transforms.Compose(
            [
                transforms.ColorJitter(brightness=.1, contrast=.1, hue=.1),
                transforms.RandomResizedCrop(size=(195, 100), scale=(0.9, 1.1), ratio=(0.4, 0.6)),
                transforms.Grayscale(),
                transforms.Resize((28, 28)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
        self.val_image_transform = transforms.Compose(
            [
                transforms.Grayscale(),
                transforms.Resize((28, 28)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])

    def data_prep(self, dataset_path="../Assets", model_output_path="../models"):
        # prepare the data and the transforms
        self.pt_path = model_output_path
        self.bs = 3
        self.train_data = FolderDataset(dataset_path, transforms=self.train_image_transform, train=True, generate_number_of_images=50)
        self.val_data = FolderDataset(dataset_path, transforms=self.val_image_transform, train=False)

        self.class_num = self.train_data.get_class_num()
        self.train_data_size = len(self.train_data)
        self.valid_data_size = len(self.val_data)
        # Create iterators for the Data loaded using DataLoader module
        self.train_data_loader = DataLoader(self.train_data, batch_size=150, shuffle=True)
        self.valid_data_loader = DataLoader(self.val_data, batch_size=self.bs, shuffle=False)

        if self.writer:
            fig = plt.figure(figsize=(15, 4))
            for idx in np.arange(16):
                ax = fig.add_subplot(2, 8, idx + 1, xticks=[], yticks=[])
                plt.imshow(im_convert(images[idx]))
                ax.set_title([labels[idx].item()])
            plt.savefig('image.png', dpi=90, bbox_inches='tight')
            plt.show()

    def model_prep(self, resnet_type=None):
        """
        Function to prepare the model
        :param resnet_type: type of resnet model
        """
        pass


