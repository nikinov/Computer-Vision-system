import torch.nn as nn
import torch
from .model_base import ModelBase
from torchvision import transforms


class PtResnet(ModelBase):
    def model_prep(self):
        self.epoch = 5
        super(PtResnet, self).model_prep()
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, self.output_size)
        self.input_size = 224
        self.model.to(self.device)
