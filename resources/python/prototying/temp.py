
from torch import nn
import torch

input_size = 784
hidden_sizes = [128, 64]
output_size = 11

model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[1], output_size),
                      nn.LogSoftmax(dim=1))

m = torch.jit.script(model.to("cuda"))

torch.jit.save(m, "model_num_naked.pt")

