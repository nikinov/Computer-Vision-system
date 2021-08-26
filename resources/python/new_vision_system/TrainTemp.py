from number_train_resnet import train

import time
import torch
import glob
import pickle
from torch import jit
import numpy


data_dir = "..\\..\\Assets5082"
tr = train(tensorboard=False, model_name='my_other_number_model')
"""
t = time.time()
# use_config=true is more performant
print(time.time() - t)
tr.prep(data_dir, "../../models")
tr.train(save_type="jit")
"""
num_incorrect = 0
num_correct = 0

lists = ['_bad_brown', '_bad_void']
lists2 = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'E']

model_path = "models/my_other_number_model.pt"
# Load data (deserialize)
model = jit.load(model_path)
for entry in glob.iglob(data_dir + '/**/*.bmp', recursive=True):
    print(entry + " Prediction: ")
    prediction = tr.predict(entry, model)
    print(prediction.clone().detach().cpu().numpy()[0])
    print(lists2.index(entry.split("\\")[-2]))
    if str(prediction.clone().detach().cpu().numpy()[0]) == str(lists2.index(entry.split("\\")[-2])):
        num_correct += 1
    else:
        num_incorrect += 1
print(" number of correct: " + str(num_correct) + " number of incorrect: " + str(num_incorrect))

""""""
#m = torch.jit.load("model/model.pt")
#print(m):