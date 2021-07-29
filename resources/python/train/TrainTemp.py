
from train import train
import time
import torch
import glob
import pickle


data_dir = "../../Assets5082"

t = time.time()
# use_config=true is more performant
tr = train(data_dir)
print(time.time() - t)
tr.data_prep(data_dir, "../../models")
tr.model_prep()
tr.train_and_validate(epochs=50, show_results=True, save="pickle")

num_incorrect = 0
num_correct = 0

model = None
# Load data (deserialize)
with open('../../models/model.pickle', 'rb') as handle:
    model = pickle.load(handle)
print(model)
for entry in glob.iglob(data_dir + '/**/*.bmp', recursive=True):
    print(entry + " Prediction: ")
    prediction = tr.predict(entry)#, model=torch.jit.load("../../models/model.pt", map_location="cuda"))
    print(prediction)
    if prediction == entry.split("\\")[-2]:
        num_correct += 1
    else:
        num_incorrect += 1
print(" number of correct: " + str(num_correct) + " number of incorrect: " + str(num_incorrect))


#m = torch.jit.load("model/model.pt")
#print(m)