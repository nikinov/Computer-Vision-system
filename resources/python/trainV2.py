#
#   find docs here:
#   https://github.com/nikinov/WickonHightech/tree/Torch
#
#   Nicholas Novelle, Jun 2021 :)
#

import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchsummary import summary
import numpy as np

import matplotlib.pyplot as plt
import os
import time
import cv2 as cv

from PIL import Image



class train:
    def __init__(self, dataset_path="../Assets", model_output_path="../models", config_file_name="data_dir_and_label_info.txt", save_config=False, use_config=False, channels=3, grayscale=False):

        splits = 80

        self.val_data = []
        self.train_data = []
        self.test_data = []

        # batch size
        self.bs = 32

        os_exists = os.path.exists(config_file_name)

        if use_config and os_exists:
            pass
        else:

            # number of classes
            self.num_classes = len(os.listdir(dataset_path))

            # 
            for j, path in enumerate(os.walk(dataset_path)):
                print(path[0])
    #def (self, model_name, num_classes, feature_extract, use_pretrained=True):


















