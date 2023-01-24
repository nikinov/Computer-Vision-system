import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from torch import nn
from torchvision import datasets, transforms
import PIL.ImageOps
import requests
from PIL import Image


def get_available_devices(n_gpu):
    sys_gpu = torch.cuda.device_count()
    if sys_gpu == 0:
        print('No GPUs detected, using the CPU')
        n_gpu = 0
    elif n_gpu > sys_gpu:
        print('Nbr of GPU requested is {} but only {} are available'.format(n_gpu, sys_gpu))
        n_gpu = sys_gpu

    device = torch.device('cuda:0' if n_gpu > 0 else 'cpu')
    print('Detected GPUs: {} Requested: {}'.format(sys_gpu, n_gpu))
    available_gpus = list(range(n_gpu))
    return device, available_gpus


def im_convert(tensor):
    image = tensor.clone().detach().numpy()
    image = image.transpose(1, 2, 0)
    image = image * np.array((0.5, 0.5, 0.5)) + np.array((0.5, 0.5, 0.5))
    image = image.clip(0, 1)
    return image


class Classifier(nn.Module):

    def __init__(self, D_in, H1, H2, D_out):
        super().__init__()
        self.linear1 = nn.Linear(D_in, H1)
        self.linear2 = nn.Linear(H1, H2)
        self.linear3 = nn.Linear(H2, D_out)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


if __name__ == "__main__":

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    device, available_gpus = get_available_devices(n_gpu=1)

    transform = transforms.Compose(
        [
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    training_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    validation_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    training_loader = torch.utils.data.DataLoader(training_dataset, batch_size=100, shuffle=True, num_workers=6, pin_memory=True)
    validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=100, shuffle=False, pin_memory=True)

    dataiter = iter(training_loader)
    images, labels = dataiter.next()

    fig = plt.figure(figsize=(25, 4))
    for idx in np.arange(20):
        ax = fig.add_subplot(2, 10, idx+1, xticks=[], yticks=[])
        plt.imshow(im_convert(images[idx]))
        ax.set_title([labels[idx].item()])
    plt.savefig('image.png', dpi=90, bbox_inches='tight')
    plt.show()    

    model = Classifier(784, 125, 65, 10)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    epochs = 10
    running_loss_history = []
    running_corrects_history = []
    val_running_loss_history = []
    val_running_corrects_history = []

    for e in range(epochs):

        running_loss = 0.0
        running_corrects = 0.0
        val_running_loss = 0.0
        val_running_corrects = 0.0

        for inputs, labels in training_loader:
            inputs = inputs.view(inputs.shape[0], -1).to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item()
            running_corrects += torch.sum(preds == labels.data)

        else:
            with torch.no_grad():
                for val_inputs, val_labels in validation_loader:
                    val_labels = val_labels.to(device)
                    val_inputs = val_inputs.view(val_inputs.shape[0], -1).to(device)
                    val_outputs = model(val_inputs)
                    val_loss = criterion(val_outputs, val_labels)

                    _, val_preds = torch.max(val_outputs, 1)
                    val_running_loss += val_loss.item()
                    val_running_corrects += torch.sum(val_preds == val_labels.data)

            epoch_loss = running_loss/len(training_loader)
            epoch_acc = running_corrects.float()/ len(training_loader)
            running_loss_history.append(epoch_loss)
            running_corrects_history.append(epoch_acc.item())

            val_epoch_loss = val_running_loss/len(validation_loader)
            val_epoch_acc = val_running_corrects.float()/ len(validation_loader)
            val_running_loss_history.append(val_epoch_loss)
            val_running_corrects_history.append(val_epoch_acc.item())
            print('epoch :', (e+1))
            print('training loss: {:.4f}, acc {:.4f} '.format(epoch_loss, epoch_acc.item()))
            print('validation loss: {:.4f}, validation acc {:.4f} '.format(val_epoch_loss, val_epoch_acc.item()))

    plt.clf()
    plt.plot(running_loss_history, label='training loss')
    plt.plot(val_running_loss_history, label='validation loss')
    plt.legend()
    plt.savefig('image_loss.png', dpi=90, bbox_inches='tight')

    plt.clf()
    plt.plot(running_corrects_history, label='training accuracy')
    plt.plot(val_running_corrects_history, label='validation accuracy')
    plt.legend()
    plt.savefig('image_acc.png', dpi=90, bbox_inches='tight')

    #exit()

    # ---------

    url = 'https://images.homedepot-static.com/productImages/007164ea-d47e-4f66-8d8c-fd9f621984a2/svn/architectural-mailboxes-house-letters-numbers-3585b-5-64_1000.jpg'
    response = requests.get(url, stream=True)
    img = Image.open(response.raw)
    plt.clf()
    plt.imshow(img)
    plt.savefig('image1.png', dpi=90, bbox_inches='tight')

    img = PIL.ImageOps.invert(img)
    img = img.convert('1')
    img = transform(img)
    plt.clf()
    plt.imshow(im_convert(img))
    plt.savefig('image2.png', dpi=90, bbox_inches='tight')

    img = img.view(img.shape[0], -1).to(device)
    output = model(img)
    _, pred = torch.max(output, 1)
    print(pred.item())

    dataiter = iter(validation_loader)
    images, labels = dataiter.next()
    images_ = images.view(images.shape[0], -1).to(device)
    output = model(images_)
    _, preds = torch.max(output, 1)

    fig = plt.figure(figsize=(25, 4))
    for idx in np.arange(20):
        ax = fig.add_subplot(2, 10, idx+1, xticks=[], yticks=[])
        plt.imshow(im_convert(images[idx]))
        ax.set_title("{} ({})".format(str(preds[idx].item()), str(labels[idx].item())), color=("green" if preds[idx]==labels[idx] else "red"))
    plt.savefig('image3.png', dpi=90, bbox_inches='tight')