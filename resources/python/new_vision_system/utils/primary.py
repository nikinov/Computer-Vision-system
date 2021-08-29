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
def run_data_to_model(data, device, model, criterion, optimizer, train=True, flat_input=False, get_prediction=False):
    grad = torch.no_grad()
    model.eval()
    if train:
        grad = torch.enable_grad()
        model.train()

    with grad:
        inputs, labels = data
        inputs = inputs.to(device)
        if flat_input:
            inputs = inputs.view(inputs.shape[0], -1)
        labels = labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        _, preds = torch.max(outputs, 1)
    if get_prediction:
        return preds
    else:
        return (loss.item(), torch.sum(preds == labels.data))

def save_model(model, type="pickle", model_name="model"):
    model.save_prep()
    if type == "jit_trace":
        m = torch.jit.trace(model.model, torch.rand(1, 3, model.get_input_size(), model.get_input_size()).to(model.device))
        m.save("models/"+model_name+".pt")
    elif type == "jit_script":
        m = torch.jit.script(model.model)
        torch.jit.save(m, "models/"+model_name+".pt")
    elif type == "pickle":
        file_handler = open("models/"+model_name+".pickle", 'wb')
        pickle.dump(model.model, file_handler)
    else:
        torch.save("models/"+model_name+".pt",model.model)
