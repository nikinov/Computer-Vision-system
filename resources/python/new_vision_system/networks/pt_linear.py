from .model_base import ModelBase
from torchvision import transforms


class PtLinearNN(ModelBase):
    def model_prep(self):
        # Parameters
        size_0 = 125
        size_1 = 65
        self.model = self.model(self.input_size*self.input_size, size_0, size_1, self.output_size).to(self.device)
    def get_train_transforms(self):
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.ColorJitter(brightness=.1, contrast=.1, hue=.1),
                transforms.RandomResizedCrop(size=(195, 100), scale=(0.9, 1.1), ratio=(0.4, 0.6)),
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