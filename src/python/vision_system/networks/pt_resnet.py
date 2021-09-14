import torch.nn as nn
import torch
from .model_base import ModelBase
from torchvision import transforms


class PtResnet(ModelBase):
    def make_input_size(self, average_input_size=None):
        return 224
    def model_prep(self, learning_rate=0.001):
        self.learning_rate = learning_rate
        super(PtResnet, self).model_prep()
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, self.output_size)
        self.model.to(self.device)
    def get_val_transforms(self):
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((224, 224)),
                transforms.Normalize((0.5,), (0.5,))
            ])
