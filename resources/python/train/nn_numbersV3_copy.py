# Imports
import torch
from torch import optim, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
import torchvision.transforms as transforms
import cv2
import glob
import numpy as np
from data_loader_test import CustomDataset
import torchvision

# Hyper parameters
input_size = 784
num_classes = 11
learning_rate = 0.01
batch_size = 1
num_epochs = 100000

# Make Network
class NN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Init transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((28, 28))
])

# Load Data

data = "../../Assets5082"

train_dataset = CustomDataset(data, transforms=transform, train=True, gray=True)
val_dataset = CustomDataset(data, transforms=transform, train=False, gray=True)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize model
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

model = NN(input_size, num_classes)#torchvision.models.resnet18(pretrained=True)#NN(input_size=input_size, num_classes=num_classes).to(device)
#set_parameter_requires_grad(model, feature_extract)
#num_ftrs = model.fc.in_features
#model.fc = nn.Linear(num_ftrs, num_classes)
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Check accuracy function
def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.reshape(x.shape[0], -1).to(device)
            #x = x.reshape(1, 3, 224, 224).to(device)
            y = y.to(device)

            output = model(x).to(device)
            _, predictions = output.max(1)
            yy = y
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print("accuracy " + str((float(num_correct)/float(num_samples))*100))

# Train Network
for epoch in range(num_epochs):
    num_right = 0
    for data, label_y in train_dataset:
        # Get to correct shape and device
        data = data.reshape(data.shape[0], -1).to(device)
        #data = data.reshape(1, 3, 224, 224).to(device)
        label = label_y.to(device)

        # forward
        model.train()
        output = model(data).to(device)
        loss = criterion(output, label)
        ll = output.max(0)
        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient decent
        optimizer.step()
    print("Epoch: " + str(epoch))
    print("train data:")
    check_accuracy(train_dataset, model)
    print("val data:")
    check_accuracy(val_dataset, model)

print("hi")



"""
for epoch in range(num_epochs):
    for batch_idx, (data, label) in enumerate(train_loader):
        data = data.to(device=device)

        print(data.shape)
"""

#train_dataset = datasets.MNIST('PATH_TO_STORE_TRAINSET', download=True, train=True, transform=transforms.ToTensor())

#print(train_dataset.data)