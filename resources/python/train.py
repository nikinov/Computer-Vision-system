#
#   find docs here:
#   https://github.com/nikinov/WickonHightech/tree/Torch
#
#   Nicholas Novelle, May 2021 :)
#

import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchsummary import summary

import matplotlib.pyplot as plt
import os
import time

from PIL import Image

class train:
    def __init__(self, dataset_path="../Assets", model_output_path="../..", save_config=False, use_config=False):
        # prepare the data and the transforms
        self.train_image_transforms = transforms.Compose([
                transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
                transforms.RandomRotation(degrees=15),
                transforms.RandomHorizontalFlip(),
                transforms.CenterCrop(size=224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])])
        self.valid_test_image_transforms =  transforms.Compose([
            transforms.Resize(size=256),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])

        os_exists = os.path.exists("data_dir_and_label_info.txt")

        split_size = {
            "train": 70,
            "val": 90,
            "test": 100
        }

        self.val_data = []
        self.train_data = []
        self.test_data = []

        # batch size
        self.bs = 32
        self.dataset_path = dataset_path
        self.pt_path = model_output_path
        self.data = []
        if use_config and os_exists:
            # get the data about the images from txt file
            f = open("data_dir_and_label_info.txt", "r")
            dataset = []
            temp_class = []
            for line in f.readlines():
                ln_split = line.split(",")
                if ln_split[2].replace('\n', '') == 'r':
                    self.train_data.append([self.train_image_transforms(Image.open(ln_split[0])), int(ln_split[1])])
                if ln_split[2].replace('\n', '') == 'v':
                    self.val_data.append([self.valid_test_image_transforms(Image.open(ln_split[0])), int(ln_split[1])])
                if ln_split[2].replace('\n', '') == 't':
                    self.train_data.append([self.valid_test_image_transforms(Image.open(ln_split[0])), int(ln_split[1])])
                if ln_split[1] not in temp_class:
                    temp_class.append(ln_split[1])

            # number of classes
            self.num_classes = len(temp_class)
        else:
            # number of classes
            self.num_classes = len(os.listdir(dataset_path))

            # Load Data from folders
            dataset = datasets.ImageFolder(dataset_path)
            self.data = dataset

            label_split = [[]] * self.num_classes
            self.train_idx = []
            self.val_idx = []
            self.test_idx = []
            labels_set = []
            print(label_split)

            # create an array of label indexes with an array of indexes
            j = 0
            for input, label in self.data:
                label_split[label] = label_split[label] + [j]
                j+=1
            # split the data into valid train and test
            # iterate over the labels or the arrays of indexes of that specific label
            for set in label_split:
                if len(set) < 10:
                    raise Exception("The number of images in one of the classes is less then 10!\nYou have to have at least 10 images per class!")
                # iterate over the indexes in the label array
                set_num = 0
                for idx in set:
                    # check if the index is in the test %
                    if len(set) - set_num > int(float(len(set))/100)*(100-split_size["train"]):
                        self.train_idx.append(idx)
                    # check if the index is in the valid %
                    elif len(set) - set_num > int(float(len(set))/100)*(100-split_size["val"]):
                        self.val_idx.append(idx)
                    # check if the index is in the test %
                    else:
                        self.test_idx.append(idx)
                    set_num += 1

            for i, (inputs, labels) in enumerate(self.data):
                if i in self.train_idx:
                    self.train_data.append([self.train_image_transforms(inputs), labels])
                    if save_config:
                        labels_set.append("r")
                elif i in self.val_idx:
                    self.val_data.append([self.valid_test_image_transforms(inputs), labels])
                    if save_config:
                        labels_set.append("v")
                else:
                    self.test_data.append([self.valid_test_image_transforms(inputs), labels])
                    if save_config:
                        labels_set.append("t")

            # save into txt if needed
            if save_config:
                if os_exists:
                    os.remove("data_dir_and_label_info.txt")
                f = open("data_dir_and_label_info.txt", "w")
                i = 0
                for im in self.data.imgs:
                    f.write(str(im).replace("'", "").replace("(", "").replace(")", "") + ',' + labels_set[i] + "\n")
                    i+=1

        # Size of Data, to be used for calculating Average Loss and Accuracy
        self.train_data_size = len(self.train_data)
        self.valid_data_size = len(self.val_data)
        self.test_data_size = len(self.test_data)

        # Create iterators for the Data loaded using DataLoader module
        self.train_data_loader = DataLoader(self.train_data, batch_size=self.bs, shuffle=True)
        self.valid_data_loader = DataLoader(self.val_data, batch_size=self.bs, shuffle=False)
        self.test_data_loader = DataLoader(self.test_data, batch_size=self.bs, shuffle=False)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.idx_to_class = {}
        i = 0
        for dir in os.listdir(dataset_path):
            self.idx_to_class[i] = dir
            i += 1
        self.resnet = models.resnet152()

    def model_prep(self, resnet_type=None, learning_rate=0.0001):
        """
        Function to prepare the model
        :param resnet_type: type of resnet model
        :param learning_rate: how much does the program learn in on go
        """

        if resnet_type == None:
            pass
        else:
            self.resnet = resnet_type

        self.resnet = self.resnet.to(self.device)

        for param in self.resnet.parameters():
            param.requires_grad = False

        # Change the final layer of ResNet Model for Transfer Learning
        self.fc_inputs = self.resnet.fc.in_features

        self.resnet.fc = nn.Sequential(
            nn.Linear(self.fc_inputs, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, self.num_classes),  # Since 2 possible outputs
            nn.LogSoftmax(dim=1)  # For using NLLLoss()
        )

        # Convert model to be used on GPU
        self.resnet = self.resnet.to(self.device)

        # Define Optimizer and Loss Function
        self.loss_func = nn.NLLLoss()
        self.optimizer = optim.Adam(self.resnet.parameters(), lr=learning_rate)

    def training_loop(self, inputs, labels, model, loss_criterion, u_loss, u_acc, train=False):
        """
        Function for iterating through the model
        :param inputs: image input
        :param labels: labels for the model
        :param model: NN model
        :param loss_criterion: loss function
        :param u_loss: loss value
        :param u_acc: accuracy value
        :param train: set if training is enabled
        :return: test loss, test accuracy, loss accuracy
        """
        grad_mode = torch.enable_grad()
        # Validation - No gradient tracking needed
        if not train:
            # Set to evaluation mode
            model.eval()
            grad_mode = torch.no_grad()
        with grad_mode:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            if train:
                # Clean existing gradients
                self.optimizer.zero_grad()
            # Forward pass - compute outputs on input data using the model
            outputs = model(inputs)

            # Compute loss
            loss = loss_criterion(outputs, labels)
            if train:
                # Backpropagate the gradients
                loss.backward()

                # Update the parameters
                self.optimizer.step()

            # Compute the total loss for the batch and add it to train_loss
            u_loss += loss.item() * inputs.size(0)

            # Compute the accuracy
            ret, predictions = torch.max(outputs.data, 1)
            correct_counts = predictions.eq(labels.data.view_as(predictions))

            # Convert correct_counts to float and then compute the mean
            acc = torch.mean(correct_counts.type(torch.FloatTensor))

            # Compute total accuracy in the whole batch and add to train_acc
            u_acc += acc.item() * inputs.size(0)

            return u_loss, u_acc, loss, acc

    def train_and_validate(self, model=None, loss_criterion=None, optimizer=None, epochs=25):
        """
        Train and validate the model
        :param model: the model to train
        :param loss_criterion: the loss function
        :param optimizer: the optimizer for the model
        :param epochs: the number of iterations
        :param show_results: plot data about the training process
        :return: a trained model with the highest accuracy
        """

        if model == None:
            model = self.resnet
        if loss_criterion == None:
            loss_criterion = self.loss_func
        if optimizer != None:
            self.optimizer = optimizer

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

            # train loop
            for inputs, labels in self.train_data_loader:
                train_loss, train_acc, loss, acc = self.training_loop(inputs, labels, model, loss_criterion, train_loss, train_acc, True)

            # Validation loop
            for inputs, labels in self.valid_data_loader:
                valid_loss, valid_acc, loss, acc = self.training_loop(inputs, labels, model, loss_criterion, valid_loss, valid_acc)

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

        summary(self.resnet, input_size=(3, 224, 224), batch_size=self.bs, device='cuda')

        # tain the model for 50 epochs
        num_epochs = 50

        return best_model

    def compute_test_set_accuracy(self, model, loss_criterion):
        '''
        Function to compute the accuracy on the test set
        Parameters
            :param model: Model to test
            :param loss_criterion: Loss Criterion to minimize
        '''

        test_acc = 0.0
        test_loss = 0.0

        # Validation loop
        j = 0
        for inputs, labels in self.test_data_loader:
            j+=1
            test_loss, test_acc, loss, acc = self.training_loop(inputs, labels, model, loss_criterion, test_loss, test_acc)
            print("Test Batch number: {:03d}, Test: Loss: {:.4f}, Accuracy: {:.4f}".format(j, loss.item(),
                                                                                           acc.item()))

        # Find average test loss and test accuracy
        avg_test_loss = test_loss / self.test_data_size
        avg_test_acc = test_acc / self.test_data_size

        print("Test accuracy : " + str(avg_test_acc))

    def predict(self, test_image_path, model=None):
        '''
        Predict the class of a single test image
        Parameters
            :param model: Model to test
            :param test_image_path: Test image path, default=trained resnet model
        '''

        if model == None:
            model = self.resnet

        transform = self.valid_test_image_transforms

        test_image = Image.open(test_image_path)
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
            cls = self.idx_to_class[topclass.cpu().numpy()[0][0]]
            score = topk.cpu().numpy()[0][0]

            for i in range(3):
                print("Predcition", i + 1, ":", self.idx_to_class[topclass.cpu().numpy()[0][i]], ", Score: ",
                      topk.cpu().numpy()[0][i])