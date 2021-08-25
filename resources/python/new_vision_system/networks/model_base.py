import torch
from torch import nn

class ModelBase():
    def __init__(self, model, device, model_name, input_size, output_size=None):
        self.input_size = input_size
        self.output_size = output_size
        self.model_name = model_name
        self.model = model
        self.epoch = 10
        self.device = device
        self.learnin_rate = 0.001
    def model_prep(self):
        self.model = self.model.to(self.device)
        pass
    def get_criterion(self):
        return nn.CrossEntropyLoss()
    def get_model_name(self):
        return self.model_name
    def get_optimizer(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.learnin_rate)
    def get_input_size(self):
        return self.input_size
    def get_train_transforms(self):
        pass
    def get_valid_transforms(self):
        pass
    def set_output_size(self, output_size):
        self.output_size = output_size


