from torchvision import transforms
from .model_base import ModelBase
from efficientnet_pytorch import EfficientNet
from src.python.vision_system.utils import math


class PtEfficientNet(ModelBase):
    def make_input_size(self, average_input_size=None):
        configs = {
            'efficientnet-b0': 224,
            'efficientnet-b1': 240,
            'efficientnet-b2': 260,
            'efficientnet-b3': 300,
            'efficientnet-b4': 380,
            'efficientnet-b5': 456,
            'efficientnet-b6': 528,
            'efficientnet-b7': 600,
            'efficientnet-b8': 672,
            'efficientnet_l2': 800,
        }
        if average_input_size is None:
            self.model_type = list(configs.keys())[0]
            return list(configs.values())[0]
        else:
            nearest = math.find_nearest(list(configs.values()), average_input_size)
            self.model_type = math.get_key(configs, nearest)
            return nearest

    def model_prep(self):
        self.model = EfficientNet.from_pretrained(self.model_type, num_classes=self.output_size).to(self.device)
    def get_val_transforms(self):
        return transforms.Compose([transforms.ToTensor(),
                                   transforms.Resize((self.input_size, self.input_size)),
                                   transforms.Normalize((0.5,), (0.5,))])
    def save_prep(self):
        self.model.set_swish(memory_efficient=False)


