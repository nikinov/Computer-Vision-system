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
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.pretrained = True
        self.models = {
            "resnet": [
                models.resnet50(pretrained=self.pretrained),
                models.resnet34(pretrained=self.pretrained),
                models.resnet18(pretrained=self.pretrained),
                models.resnet152(pretrained=self.pretrained),
                models.resnet101(pretrained=self.pretrained),
                models.resnext101_32x8d(pretrained=self.pretrained),
                models.resnext50_32x4d(pretrained=self.pretrained),
                models.wide_resnet101_2(pretrained=self.pretrained),
                models.wide_resnet50_2(pretrained=self.pretrained)
            ],
            "alexnet": [
                models.alexnet(pretrained=self.pretrained)
            ],
            "vgg": [
                models.vgg16(pretrained=self.pretrained),
                models.vgg11(pretrained=self.pretrained),
                models.vgg13(pretrained=self.pretrained),
                models.vgg19(pretrained=self.pretrained),
                models.vgg16_bn(pretrained=self.pretrained),
                models.vgg11_bn(pretrained=self.pretrained),
                models.vgg13_bn(pretrained=self.pretrained),
                models.vgg19_bn(pretrained=self.pretrained)
            ],
            "squeezenet": [
                models.squeezenet1_0(pretrained=self.pretrained),
                models.squeezenet1_1(pretrained=self.pretrained)
            ],
            "densenet": [
                models.densenet121(pretrained=self.pretrained),
                models.densenet161(pretrained=self.pretrained),
                models.densenet169(pretrained=self.pretrained),
                models.densenet201(pretrained=self.pretrained)
            ],
            "inception": [
                models.inception_v3(pretrained=self.pretrained)
            ],
            "googlenet": [
                models.googlenet(pretrained=self.pretrained)
            ],
            "mnasnet": [
                models.mnasnet0_5(pretrained=self.pretrained),
                models.mnasnet1_0(pretrained=self.pretrained)
            ],
            "mobilenet": [
                models.mobilenet_v2(pretrained=self.pretrained)
            ]
        }

    def set_parameter_requires_grad(self, model, feature_extracting):
        if feature_extracting:
            for param in model.parameters():
                param.requires_grad = False

    def model_prep(self, model_name, model_ft, feature_extract):
        if model_name == "resnet":
            """ Resnet
            """
            self.set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, self.num_classes)
            self.input_size = 224

        elif model_name == "alexnet":
            """ Alexnet
            """
            self.set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.classifier[6].in_features
            model_ft.classifier[6] = nn.Linear(num_ftrs, self.num_classes)
            self.input_size = 224

        elif model_name == "vgg":
            """ VGG
            """
            self.set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.classifier[6].in_features
            model_ft.classifier[6] = nn.Linear(num_ftrs, self.num_classes)
            self.input_size = 224

        elif model_name == "squeezenet":
            """ Squeezenet
            """
            self.set_parameter_requires_grad(model_ft, feature_extract)
            model_ft.classifier[1] = nn.Conv2d(512, self.num_classes, kernel_size=(1, 1), stride=(1, 1))
            model_ft.num_classes = self.num_classes
            self.input_size = 224

        elif model_name == "densenet":
            """ Densenet
            """
            self.set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.classifier.in_features
            model_ft.classifier = nn.Linear(num_ftrs, self.num_classes)
            self.input_size = 224

        elif model_name == "inception":
            """ Inception v3
            Be careful, expects (299,299) sized images and has auxiliary output
            """
            self.set_parameter_requires_grad(model_ft, feature_extract)
            # Handle the auxilary net
            num_ftrs = model_ft.AuxLogits.fc.in_features
            model_ft.AuxLogits.fc = nn.Linear(num_ftrs, self.num_classes)
            # Handle the primary net
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, self.num_classes)
            self.input_size = 299
        elif model_name == "googlenet":
            self.set_parameter_requires_grad(model_ft, feature_extract)

            num_ftrs = model_ft.AuxLogits.fc.in_features
            model_ft.AuxLogits.fc = nn.Linear(num_ftrs, self.num_classes)

            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, self.num_classes)

        else:
            model_ft = None
        return model_ft


















