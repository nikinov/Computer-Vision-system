from train import AI
from networks import pt_linear, pt_resnet, pt_inception, custom_networks, pt_efficient_net
from torchvision import models
from data_preprocessing import transforms
from utils import primary

import time
import torch
import glob
import pickle
from torch import jit
import numpy
from data_preprocessing.transforms import MyTransforms


data_dir = "..\\..\\Assets"
tr = AI()

t = time.time()
# use_config=true is more performant
print(time.time() - t)
my_t = MyTransforms()
tr.prep(data_dir, "../../models", train_trans=my_t.get_train_transforms(tr.model.input_size), batch_size=50, generate_images_per_image=50)
tr.train(save_type="jit_trace", epochs=7)

model_path = "models/the_new_guy.pt"
# Load data (deserialize)
model = jit.load(model_path)

num_correct, num_incorrect = tr.predict_folder(data_dir, pt_efficient_net.PtEfficientNet(model, model_name="some_random_name"))

print(" number of correct: " + str(num_correct) + " number of incorrect: " + str(num_incorrect))

""""""
#m = torch.jit.load("model/model.pt")
#print(m):