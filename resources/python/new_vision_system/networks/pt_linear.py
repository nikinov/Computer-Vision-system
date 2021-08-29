from torchvision import transforms
from .model_base import ModelBase


class PtLinearNN(ModelBase):
    def make_input_size(self, average_input_size=None):
        return 28
    def model_prep(self):
        # Parameters
        size_0 = 125
        size_1 = 65
        self.model = self.model(self.input_size*self.input_size, size_0, size_1, self.output_size).to(self.device)
    def get_val_transforms(self):
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Grayscale(),
                transforms.Resize((self.input_size, self.input_size)),
                transforms.Normalize((0.5,), (0.5,))
            ])
