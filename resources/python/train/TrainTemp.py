
from train import train
import time
import torch
import glob


data_dir = "../../Assets5082"

t = time.time()
# use_config=true is more performant
tr = train(dataset_path=data_dir, model_output_path="models")
print(time.time() - t)
tr.model_prep()

tr.train_and_validate(epochs=50, show_results=True)

num_incorrect = 0
num_correct = 0

for entry in glob.iglob(data_dir + '/**/*.bmp', recursive=True):
    print(entry + " Prediction: ")
    prediction = tr.predict(entry)
    if prediction == entry.split("/")[-2]:
        num_correct += 1
    else:
        num_incorrect += 1
print(" number of correct: " + str(num_correct) + " number of incorrect: " + str(num_incorrect))


#m = torch.jit.load("model/model.pt")
#print(m)