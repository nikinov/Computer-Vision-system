from fastai.vision import *
from DataLoaders import FolderDataset
import fastai.datasets

dataset_path = "../../Assets5082"

dls = ImageDataBunch.from_folder(dataset_path, )

learner = cnn_learner()