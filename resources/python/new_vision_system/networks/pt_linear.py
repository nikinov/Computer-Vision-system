from .model_base import ModelBase
from torchvision import transforms


class PtLinearNN(ModelBase):
    def model_prep(self):
        # Parameters
        size_0 = 125
        size_1 = 65
        self.model = self.model(self.input_size*self.input_size, size_0, size_1, self.output_size).to(self.device)
