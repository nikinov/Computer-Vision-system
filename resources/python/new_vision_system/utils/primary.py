#
#
#
#
# Nicholas Novelle July 2021
#

import torch

import numpy as np
import pickle


# will convert tensor into image
def im_convert(tensor):
    image = tensor.clone().detach().numpy()
    image = image.transpose(1, 2, 0)
    image = image * np.array((0.5, 0.5, 0.5)) + np.array((0.5, 0.5, 0.5))
    image = image.clip(0, 1)
    return image

# prepare data for forward pass
def run_data_to_model(data, device, model, criterion, optimizer, train=True):
    grad = torch.no_grad()
    model.eval()
    if train:
        grad = torch.enable_grad()
        model.train()

    with grad:
        inputs, labels = data
        inputs = inputs.view(inputs.shape[0], -1).to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        _, preds = torch.max(outputs, 1)
    return (loss.item(), torch.sum(preds == labels.data))

def save_model(model, type="pickle"):
    if type == "jit":
        m = torch.jit.script(model)
        torch.jit.save("models/jit_model.pt",m)
    elif type == "pickle":
        filehandler = open("models/pickle_model.pt", 'wb')
        pickle.dump(model, filehandler, pickle.HIGHEST_PROTOCOL)
    else:
        torch.save("models/model.pt",model)
