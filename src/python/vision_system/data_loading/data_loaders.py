#
#
#
#
# Nicholas Novelle July 2021
#

from torchvision import transforms
import torch
from torch.utils.data import Dataset

from skimage import io
import os
import glob
import cv2
import pandas as pd
from functools import reduce


# loads data from folders and automatically allocates classes
class FolderDataset(Dataset):
    def __init__(self, data_dir, transforms=None, train=True, color=False, train_split=0.8, generate_number_of_images=1):
        self.color = color
        # important double check the directory separator
        self.separator = "\\"
        self.annotations = []
        self.input_sizes = []
        # class split looks like this {name_of_cass:list_of_paths}
        self.class_track = {}
        for entry in glob.iglob(data_dir + '/**/*.bmp', recursive=True):
            temp = entry.split(self.separator)[-2]
            if temp in self.class_track.keys():
                self.class_track[temp].append(entry)
            else:
                self.class_track[temp] = []

        self.smallest_class_num = len(list(self.class_track.values())[0])
        for n in list(self.class_track.values()):
            self.smallest_class_num = min(self.smallest_class_num, len(n))
        for ls in list(self.class_track.values()):
            for i in range(len(ls) - self.smallest_class_num):
                ls.pop()
            for i, el in enumerate(ls):
                if train and i < self.smallest_class_num*train_split:
                    for i in range(generate_number_of_images):
                        self.annotations.append(el)
                elif i > self.smallest_class_num*train_split and not train:
                    self.annotations.append(el)
                # add width and height to the input_sizes used for finding the right model
                width, height, _ = io.imread(el).shape
                self.input_sizes.append(width)
                self.input_sizes.append(height)
        self.data_dir = data_dir
        self.transforms = transforms

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, item):
        img_path = os.path.join(self.annotations[item])
        if self.color:
            image = io.imread(img_path)
        else:
            image = cv2.imread(img_path, 0)
        y_label = torch.tensor(int(list(self.class_track.keys()).index(self.annotations[item].split(self.separator)[-2]))) # problem make forloop

        if self.transforms:
            image = self.transforms(image)
        return (image, y_label)

    def get_class_num(self):
        return len(list(self.class_track.keys()))

    def get_optimal_input_size(self):
        return reduce(lambda a, b: a + b, self.input_sizes) / len(self.input_sizes)

    def set_transforms(self, trans):
        self.transforms = trans

# this loads data from the csv file
class CSVDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, gray=False, mode='trn'):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.gray = gray
        self.mode = mode
        for index in range(self.annotations.shape[0]):
            if self.annotations.iloc[index, 2] != self.mode:
                self.annotations.drop(index)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, item):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[item, 2])
        if self.gray:
            image = cv2.imread(img_path, 0)
        else:
            image = io.imread(img_path)
        y_label = torch.tensor(int(self.annotations.iloc[item, 1]))

        if self.transform:
            image = self.transform(image)

        return (image, y_label)


