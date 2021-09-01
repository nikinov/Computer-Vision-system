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

# save model
def save_model(model, my_type="pickle", model_name="model", output_path="models/"):
    model.save_prep()
    if my_type == "jit_trace":
        m = torch.jit.trace(model.model, torch.rand(1, 3, model.get_input_size(), model.get_input_size()).to(model.device))
        m.save(output_path+model_name+".pt")
    elif my_type == "jit_script":
        m = torch.jit.script(model.model)
        torch.jit.save(m, output_path+model_name+".pt")
    elif my_type == "pickle":
        file_handler = open(output_path+model_name+".pickle", 'wb')
        pickle.dump(model.model, file_handler)
    else:
        torch.save(output_path+model_name+".pt",model.model)

# load model
def load_model(model_path, my_type):
    if my_type == "jit":
        return torch.jit.load(model_path)
    elif my_type == "pickle":
        f = open(model_path, 'rb')
        return pickle.load(f)
