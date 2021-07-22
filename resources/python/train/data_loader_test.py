import torch
from skimage import io
from torch.utils.data import Dataset
import os
import glob


class CustomDataset(Dataset):
    def __init__(self, data_dir, transforms=None, train=True, train_split=0.8):
        self.annotations = []
        self.class_track = {}
        for entry in glob.iglob(data_dir + '/**/*.bmp', recursive=True):
            temp = entry.split("/")[-2]
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
                    self.annotations.append(el)
                elif i > self.smallest_class_num*train_split and not train:
                    self.annotations.append(el)
        self.data_dir = data_dir
        self.transforms = transforms

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, item):
        img_path = os.path.join(self.annotations[item])
        image = io.imread(img_path)
        y_label = torch.tensor([int(list(self.class_track.keys()).index(self.annotations[item].split("/")[-2]))])

        if self.transforms:
            image = self.transforms(image)
        return (image, y_label)




