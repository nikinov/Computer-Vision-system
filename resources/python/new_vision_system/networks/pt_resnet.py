import torch.nn as nn
import torch
from .model_base import ModelBase
from torchvision import transforms


class PtResnet(ModelBase):
    def model_prep(self):
        super(PtResnet, self).model_prep()
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, self.output_size)
        self.input_size = 224
    def get_train_transforms(self):
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.ColorJitter(brightness=.1, contrast=.1, hue=.1),
                transforms.RandomResizedCrop(size=(self.input_size, self.input_size), scale=(0.9, 1.1), ratio=(0.4, 0.6)),
                transforms.Grayscale(),
                transforms.Resize((self.input_size, self.input_size)),
                transforms.Normalize((0.5,), (0.5,))
            ])
    def get_valid_transforms(self):
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Grayscale(),
                transforms.Resize((self.input_size, self.input_size)),
                transforms.Normalize((0.5,), (0.5,))
            ])
