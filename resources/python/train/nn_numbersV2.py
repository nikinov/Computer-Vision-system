from torch import nn
import os
import glob

directory = "C:/!SAMPLES!/1716-5082/Assets5082-samples-for-training/Assets5082"

for entry in glob.iglob(directory + '**/**', recursive=True):
    print(entry)