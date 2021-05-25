from train import train
from train_c import train as train2

tr = train()

tr.model_prep()

tr.train_and_validate()
