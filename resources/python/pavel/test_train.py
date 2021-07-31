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

from DataSampler import DataSampler

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

    def predict(self, inputs):
        outputs = self(inputs)
        _, preds = torch.max(outputs, 1)
        return preds

    @staticmethod
    def loss_fnc():
        return nn.CrossEntropyLoss()

def model_save(model, path):

    torch.save(model.state_dict(), path)


def model_load(path, device):

    model = Classifier(28*28, 125, 65, 11)
    model.to(device)
    model.load_state_dict(torch.load(path))
    model.eval()

    return model


def model_train(device, training_loader, validation_loader):

    model = Classifier(28*28, 125, 65, 11)
    model.to(device)

    criterion = model.loss_fnc()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print('Training ... ')
    epochs = 10
    running_loss_history = []
    running_corrects_history = []
    val_running_loss_history = []
    val_running_corrects_history = []

    for e in range(epochs):

        running_loss = 0.0
        running_corrects = 0.0
        num_train = 0
        val_running_loss = 0.0
        val_running_corrects = 0.0
        num_val = 0

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
            num_train += len(labels)

        else:
            with torch.no_grad():
                for val_inputs, val_labels in validation_loader:
                    val_labels = val_labels.to(device)
                    val_inputs = val_inputs.view(val_inputs.shape[0], -1).to(device)
                    val_outputs = model(val_inputs)
                    val_loss = criterion(val_outputs, val_labels)

                    val_preds = model.predict(val_inputs)

                    val_running_loss += val_loss.item()
                    val_running_corrects += torch.sum(val_preds == val_labels.data)
                    num_val += len(val_labels)

            epoch_loss = running_loss/num_train
            epoch_acc = running_corrects.item()/num_train
            running_loss_history.append(epoch_loss)
            running_corrects_history.append(epoch_acc)

            val_epoch_loss = val_running_loss/num_val
            val_epoch_acc = val_running_corrects.item()/num_val
            val_running_loss_history.append(val_epoch_loss)
            val_running_corrects_history.append(val_epoch_acc)
            print('epoch :', (e+1))
            print('training loss: {:.4f}, acc {:.4f} '.format(epoch_loss, epoch_acc))
            print('validation loss: {:.4f}, validation acc {:.4f} '.format(val_epoch_loss, val_epoch_acc))

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

    return model

def main():

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    device, available_gpus = get_available_devices(n_gpu=1)

    training_transform = transforms.Compose(
        [
            transforms.ColorJitter(brightness=.1, contrast=.1, hue=.1),
            transforms.RandomResizedCrop(size=(195,100), scale=(0.9,1.1), ratio=(0.4,0.6)),
            transforms.Grayscale(),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    validation_transform = transforms.Compose(
        [
            transforms.Grayscale(),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    # load trn/val data splits from file
    dataloader = DataSampler(
        path='../../Assets5082',
        labels={'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 'E': 10},
        filename='list.csv')

    print('Preparing trn/val data ...')
    training_dataset = []
    for label, image in dataloader.get_label_image('trn'):
        for _ in range(100):
            training_dataset.append([training_transform(image), label])

    validation_dataset = []
    for label, image in dataloader.get_label_image('val'):
        validation_dataset.append([validation_transform(image), label])

    print('Init data loader ...')
    training_loader = torch.utils.data.DataLoader(training_dataset, batch_size=64, shuffle=True, num_workers=6, pin_memory=True)
    validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=16, shuffle=False, pin_memory=True)

    dataiter = iter(training_loader)
    images, labels = dataiter.next()
    fig = plt.figure(figsize=(15, 4))
    for idx in np.arange(16):
        ax = fig.add_subplot(2, 8, idx+1, xticks=[], yticks=[])
        plt.imshow(im_convert(images[idx]))
        ax.set_title([labels[idx].item()])
    plt.savefig('image.png', dpi=90, bbox_inches='tight')
    plt.show()

    model = model_train(device, training_loader, validation_loader)
    model_save(model, "model.bin")
    #model = model_load("model.bin", device)

    print('Predict ...')
    for images, labels in validation_loader:

        images_ = images.view(images.shape[0], -1).to(device)
        output = model(images_)
        _, preds = torch.max(output, 1)
        for lab, pred  in zip(labels, preds):
            print(f'gt: {lab.item()}      pred: {pred.item()}  {"x" if lab.item() != pred.item() else ""}')

if __name__ == "__main__":
    main()
    print('done')
