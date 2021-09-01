import torch
from torch import nn
from torchvision import transforms
from torchvision.models.resnet import ResNet, BasicBlock


class ModelBase():
    def __init__(self, model=None, model_name="my_model"):
        self.input_size = self.make_input_size()
        self.model_name = model_name
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.learning_rate = 0.001
    def make_input_size(self, average_input_size=None):
        pass
    def model_prep(self):
        self.model = self.model.to(self.device)
        pass
    def save_prep(self):
        pass
    def get_criterion(self):
        return nn.CrossEntropyLoss()
    def get_model_name(self):
        return self.model_name
    def get_optimizer(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
    def get_input_size(self):
        return self.input_size
    def set_output_size(self, output_size):
        self.output_size = int(output_size)
    def get_val_transforms(self):
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((self.input_size, self.input_size)),
                transforms.Normalize((0.5,), (0.5,))
            ])




