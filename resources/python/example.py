from train import train
import torch

tr = train()

tr.model_prep()

tr.train_and_validate(show_results=True)
