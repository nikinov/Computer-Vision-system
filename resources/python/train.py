import torch, torchvision
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
from torchsummary import summary

import numpy as np
import matplotlib.pyplot as plt
import os

from PIL import Image


class train:
    def __init__(self):
        # prepare the data and the transforms
        self.image_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
                transforms.RandomRotation(degrees=15),
                transforms.RandomHorizontalFlip(),
                transforms.CenterCrop(size=224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
            ]),
            'valid': transforms.Compose([
                transforms.Resize(size=256),
                transforms.CenterCrop(size=224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize(size=256),
                transforms.CenterCrop(size=224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
            ])
        }
        self.dataset_path = "../Assets"
        self.pt_path = "../models"

        # create train valid and test directory
        train_directory = os.path.join(self.dataset_path, 'train')
        valid_directory = os.path.join(self.dataset_path, 'val')
        test_directory = os.path.join(self.dataset_path, 'test')

        # batch size
        self.bs = 32

        # number of classes
        self.num_classes = len(os.listdir(valid_directory))

        # Load Data from folders
        self.data = {
            'train': datasets.ImageFolder(root=train_directory, transform=self.image_transforms['train']),
            'valid': datasets.ImageFolder(root=valid_directory, transform=self.image_transforms['valid']),
            'test': datasets.ImageFolder(root=test_directory, transform=self.image_transforms['test'])
        }

        # Size of Data, to be used for calculating Average Loss and Accuracy
        self.train_data_size = len(data['train'])
        self.valid_data_size = len(data['valid'])
        self.test_data_size = len(data['test'])

        # Create iterators for the Data loaded using DataLoader module
        self.train_data_loader = DataLoader(data['train'], batch_size=self.bs, shuffle=True)
        self.valid_data_loader = DataLoader(data['valid'], batch_size=self.bs, shuffle=True)
        self.test_data_loader = DataLoader(data['test'], batch_size=self.bs, shuffle=True)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def model_prep(self, resnet_type=models.resnet152):
        """
        Function to prepare the model
        :param resnet_type: type of resnet model
        :return:
        """
        self.resnet = resnet_type

        self.resnet = self.resnet.to(device)

        for param in self.resnet.parameters():
            param.requires_grad = False

        # Change the final layer of ResNet Model for Transfer Learning
        self.fc_inputs = self.resnet.fc.in_features

        self.resnet.fc = nn.Sequential(
            nn.Linear(fc_inputs, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, self.num_classes),  # Since 2 possible outputs
            nn.LogSoftmax(dim=1)  # For using NLLLoss()
        )

        # Convert model to be used on GPU
        self.resnet = self.resnet.to(self.device)

        # Define Optimizer and Loss Function
        self.loss_func = nn.NLLLoss()
        self.optimizer = optim.Adam(self.resnet.parameters())

    # train and validate the model
    def train_and_validate(self, model, loss_criterion, optimizer, epochs=25):
        '''
        Function to train and validate
        Parameters
            :param model: Model to train and validate
            :param loss_criterion: Loss Criterion to minimize
            :param optimizer: Optimizer for computing gradients
            :param epochs: Number of epochs (default=25)

        Returns
            model: Trained Model with best validation accuracy
            history: (dict object): Having training loss, accuracy and validation loss, accuracy
        '''

        start = time.time()
        history = []
        best_loss = 100000.0
        best_epoch = None
        best_model = model
        for epoch in range(epochs):
            epoch_start = time.time()
            print("Epoch: {}/{}".format(epoch + 1, epochs))

            # Set to training mode
            model.train()

            # Loss and Accuracy within the epoch
            train_loss = 0.0
            train_acc = 0.0

            valid_loss = 0.0
            valid_acc = 0.0

            for i, (inputs, labels) in enumerate(self.train_data_loader):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                # Clean existing gradients
                optimizer.zero_grad()

                # Forward pass - compute outputs on input data using the model
                outputs = model(inputs)

                # Compute loss
                loss = loss_criterion(outputs, labels)

                # Backpropagate the gradients
                loss.backward()

                # Update the parameters
                optimizer.step()

                # Compute the total loss for the batch and add it to train_loss
                train_loss += loss.item() * inputs.size(0)

                # Compute the accuracy
                ret, predictions = torch.max(outputs.data, 1)
                correct_counts = predictions.eq(labels.data.view_as(predictions))

                # Convert correct_counts to float and then compute the mean
                acc = torch.mean(correct_counts.type(torch.FloatTensor))

                # Compute total accuracy in the whole batch and add to train_acc
                train_acc += acc.item() * inputs.size(0)

                # print("Batch number: {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}".format(i, loss.item(), acc.item()))

            # Validation - No gradient tracking needed
            with torch.no_grad():

                # Set to evaluation mode
                model.eval()

                # Validation loop
                for j, (inputs, labels) in enumerate(self.valid_data_loader):
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    # Forward pass - compute outputs on input data using the model
                    outputs = model(inputs)

                    # Compute loss
                    loss = loss_criterion(outputs, labels)

                    # Compute the total loss for the batch and add it to valid_loss
                    valid_loss += loss.item() * inputs.size(0)

                    # Calculate validation accuracy
                    ret, predictions = torch.max(outputs.data, 1)
                    correct_counts = predictions.eq(labels.data.view_as(predictions))

                    # Convert correct_counts to float and then compute the mean
                    acc = torch.mean(correct_counts.type(torch.FloatTensor))

                    # Compute total accuracy in the whole batch and add to valid_acc
                    valid_acc += acc.item() * inputs.size(0)

                    # print("Validation Batch number: {:03d}, Validation: Loss: {:.4f}, Accuracy: {:.4f}".format(j, loss.item(), acc.item()))
            if valid_loss < best_loss:
                best_loss = valid_loss
                best_epoch = epoch
                best_model = model

            # Find average training loss and training accuracy
            avg_train_loss = train_loss / self.train_data_size
            avg_train_acc = train_acc / self.train_data_size

            # Find average training loss and training accuracy
            avg_valid_loss = valid_loss / self.valid_data_size
            avg_valid_acc = valid_acc / self.valid_data_size

            history.append([avg_train_loss, avg_valid_loss, avg_train_acc, avg_valid_acc])

            epoch_end = time.time()

            print(
                "Epoch : {:03d}, Training: Loss - {:.4f}, Accuracy - {:.4f}%, \n\t\tValidation : Loss - {:.4f}, Accuracy - {:.4f}%, Time: {:.4f}s".format(
                    epoch, avg_train_loss, avg_train_acc * 100, avg_valid_loss, avg_valid_acc * 100,
                                           epoch_end - epoch_start))

        # save the model with the best accuracy
        best_model = torch.jit.script(best_model)
        torch.jit.save(best_model, self.pt_path + '/model' + '.pt')

        return best_model, history, best_epoch

    def computeTestSetAccuracy(self, model, loss_criterion):
        '''
        Function to compute the accuracy on the test set
        Parameters
            :param model: Model to test
            :param loss_criterion: Loss Criterion to minimize
        '''

        test_acc = 0.0
        test_loss = 0.0

        # Validation - No gradient tracking needed
        with torch.no_grad():
            # Set to evaluation mode
            model.eval()

            # Validation loop
            for j, (inputs, labels) in enumerate(self.test_data_loader):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                # Forward pass - compute outputs on input data using the model
                outputs = model(inputs)

                # Compute loss
                loss = loss_criterion(outputs, labels)

                # Compute the total loss for the batch and add it to valid_loss
                test_loss += loss.item() * inputs.size(0)

                # Calculate validation accuracy
                ret, predictions = torch.max(outputs.data, 1)
                correct_counts = predictions.eq(labels.data.view_as(predictions))

                # Convert correct_counts to float and then compute the mean
                acc = torch.mean(correct_counts.type(torch.FloatTensor))

                # Compute total accuracy in the whole batch and add to valid_acc
                test_acc += acc.item() * inputs.size(0)

                print("Test Batch number: {:03d}, Test: Loss: {:.4f}, Accuracy: {:.4f}".format(j, loss.item(),
                                                                                               acc.item()))

        # Find average test loss and test accuracy
        avg_test_loss = test_loss / self.test_data_size
        avg_test_acc = test_acc / self.test_data_size

        print("Test accuracy : " + str(avg_test_acc))

    def predict(self, model, test_image_name):
        '''
        Function to predict the class of a single test image
        Parameters
            :param model: Model to test
            :param test_image_name: Test image

        '''

        transform = self.image_transforms['test']

        test_image = Image.open(test_image_name)
        plt.imshow(test_image)

        test_image_tensor = transform(test_image)
        if torch.cuda.is_available():
            test_image_tensor = test_image_tensor.view(1, 3, 224, 224).cuda()
        else:
            test_image_tensor = test_image_tensor.view(1, 3, 224, 224)

        with torch.no_grad():
            model.eval()
            # Model outputs log probabilities
            out = model(test_image_tensor)
            ps = torch.exp(out)

            topk, topclass = ps.topk(3, dim=1)
            cls = idx_to_class[topclass.cpu().numpy()[0][0]]
            score = topk.cpu().numpy()[0][0]

            for i in range(3):
                print("Predcition", i + 1, ":", idx_to_class[topclass.cpu().numpy()[0][i]], ", Score: ",
                      topk.cpu().numpy()[0][i])

    def make_model(self, show_results=False, asset_path=self.dataset_path):
        """
        Function to make a model with an option to show data about it
        :param show_results: to show more info about the training process
        :param asset_path: the path to assets for training the model
        :return: the trained model from the best performing epoch
        """
        self.dataset_path = asset_path
        summary(self.resnet, input_size=(3, 224, 224), batch_size=self.bs, device='cuda')

        # tain the model for 50 epochs
        num_of_epochs = 50
        trained_model, history, best_epoch = train_and_validate(self.resnet, self.loss_func, self.optimizer, num_epochs)

        if showresults:
            plt.plot(history[:, 0:2])
            plt.legend(['Tr Loss', 'Val Loss'])
            plt.xlabel('Epoch Number')
            plt.ylabel('Loss')
            plt.ylim(0, 1)
            plt.savefig(pt_save + '_loss_curve.png')
            plt.show()

            plt.plot(history[:, 2:4])
            plt.legend(['Tr Accuracy', 'Val Accuracy'])
            plt.xlabel('Epoch Number')
            plt.ylabel('Accuracy')
            plt.ylim(0, 1)
            plt.savefig(pt_save + '_accuracy_curve.png')
            plt.show()

        return trained_model