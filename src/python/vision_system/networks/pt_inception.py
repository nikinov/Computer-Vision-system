from .model_base import ModelBase
from torch import nn
from torchvision import transforms


class PtInspection(ModelBase):
    def make_input_size(self, average_input_size=None):
        return 299
    def model_prep(self):
        # Handle the auxilary net
        num_ftrs = self.model.AuxLogits.fc.in_features
        self.model.AuxLogits.fc = nn.Linear(num_ftrs, self.output_size)
        # Handle the primary net
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, self.output_size)
    def get_val_transforms(self):
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((self.input_size, self.input_size)),
                transforms.Normalize((0.5,), (0.5,))
            ])