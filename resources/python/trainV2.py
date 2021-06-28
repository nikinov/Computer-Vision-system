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
from pathlib import Path



class train:
    def __init__(self, dataset_path="../Assets", model_output_path="../models", config_file_name="data_dir_and_label_info.txt", save_config=False, use_config=False, channels=3, grayscale=False):

        splits = 80

        self.data = [[]]

        self.val_data = []
        self.train_data = []
        self.test_data = []

        # batch size
        self.bs = 32
        self.data_path = dataset_path
        self.grayscale = grayscale
        os_exists = os.path.exists(config_file_name)

        if use_config and os_exists:

            pass
        else:
            # number of classes
            self.num_classes = len(os.listdir(dataset_path))

            # load data from folders
            if grayscale:
                self.load_data(dataset_path)


            else:

                # number of classes
                self.num_classes = len(os.listdir(dataset_path))

                #
                for j, path in enumerate(os.walk(dataset_path)):
                    print(path[0])

    def load_data(self, dirs):
        for dr in os.listdir(dirs):
            if not dr.endswith(".DS_Store"):
                if dr.endswith(".bmp") or dr.endswith(".png"):
                    arr = [self.transform_with_cv(Image.open(os.path.join(dirs, dr))), Path(dirs).parent.absolute().name]
                    self.data.append(arr)
                else:
                    self.load_data(os.path.join(dirs, dr))

    def transform_with_cv(self, im):
        return im
"""
    def model_prep(self):

    def training_loop(self):

    def train_and_validate(self):

    def compute_test_set_accuracy(self):

    def predict(self):


    """
    #def (self, model_name, num_classes, feature_extract, use_pretrained=True):


















