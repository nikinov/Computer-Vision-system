import torch
from torch.utils.data import DataLoader
from networks import pt_efficient_net

import matplotlib.pyplot as plt
import numpy as np

from data_loading.data_loaders import FolderDataset
from utils.primary import im_convert, run_data_to_model, save_model
from utils.visualisation import print_metrix, plot_metrix
from data_preprocessing.transforms import MyTransforms
from skimage import io
import glob


class AI():
    def __init__(self, model=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if model is None:
            self.model = pt_efficient_net.PtEfficientNet(None, model_name="my_model")
        else:
            self.model = model
        self.tr = MyTransforms()

    def prep(self, dataset_path="../Assets", model_output_path="../models", learning_rate=0.001, batch_size=5, generate_images_per_image=40, train_trans=None, visualisation=False):
        # params
        self.pt_path = model_output_path
        self.bs = batch_size
        self.model.learning_rate = learning_rate

        # default transforms
        if train_trans is None:
            train_trans = self.tr.get_train_transforms(self.model.input_size)

        # data and model prep
        self.val_data = FolderDataset(dataset_path, transforms=None, train=False, color=True)
        self.model.make_input_size(self.val_data.get_optimal_input_size())
        self.val_data.set_transforms(self.model.get_val_transforms())
        self.train_data = FolderDataset(dataset_path, transforms=train_trans, train=True, generate_number_of_images=generate_images_per_image, color=True)

        self.class_num = self.train_data.get_class_num()

        self.model.set_output_size(self.class_num)
        self.model.model_prep()

        self.train_data_size = len(self.train_data)
        self.valid_data_size = len(self.val_data)
        # Create iterators for the Data loaded using DataLoader module
        self.train_data_loader = DataLoader(self.train_data, batch_size=self.bs, shuffle=True)
        self.valid_data_loader = DataLoader(self.val_data, 1, shuffle=False)

        # looks
        if visualisation:
            dataset = iter(self.train_data_loader)
            images, labels = next(dataset)
            fig = plt.figure(figsize=(15, 4))
            for idx in np.arange(16):
                ax = fig.add_subplot(2, 8, idx + 1, xticks=[], yticks=[])
                plt.imshow(im_convert(images[idx]))
                ax.set_title([labels[idx].item()])
            plt.savefig('image.png', dpi=90, bbox_inches='tight')
            plt.show()

    def train(self, save_type="None", epochs=5, flat=False):
        self.running_loss_history = []
        self.running_corrects_history = []
        self.val_running_loss_history = []
        self.val_running_corrects_history = []

        for e in range(epochs):
            # parameters
            running_loss = 0.0
            running_corrects = 0.0
            running_loss_val = 0.0
            running_corrects_val = 0.0

            for i, data in enumerate(self.train_data_loader):
                loss, corrects = run_data_to_model(data, self.device, self.model.model, self.model.get_criterion(), self.model.get_optimizer(), train=True, flat_input=flat)
                running_loss += loss
                running_corrects += corrects
            for i, data in enumerate(self.valid_data_loader):
                loss, corrects = run_data_to_model(data, self.device, self.model.model, self.model.get_criterion(), self.model.get_optimizer(), train=False, flat_input=flat)
                running_loss_val += loss
                running_corrects_val += corrects
            epoch_loss = running_loss/len(self.train_data_loader)
            epoch_acc = running_corrects.item()/len(self.train_data_loader)/self.bs
            self.running_loss_history.append(epoch_loss)
            self.running_corrects_history.append(epoch_acc)

            val_epoch_loss = running_loss_val/len(self.valid_data_loader)
            val_epoch_acc = running_corrects_val.item()/len(self.valid_data_loader)
            self.val_running_loss_history.append(val_epoch_loss)
            self.val_running_corrects_history.append(val_epoch_acc)
            print_metrix(e, epoch_loss, epoch_acc, val_epoch_loss, val_epoch_acc)

        plot_metrix(self.running_loss_history, self.val_running_loss_history, self.running_corrects_history, self.val_running_corrects_history)
        torch.cuda.empty_cache()
        if save_type == "None":
            pass
        else:
            save_model(model=self.model, my_type=save_type, model_name=self.model.get_model_name())
        return self.model

    def predict_folder(self, folder_path, model=None, flat=False, reshape=True):
        num_incorrect = 0
        num_correct = 0
        listt = []
        if model is None:
            model = self.model
        for entry in glob.iglob(f'{folder_path}/*'):
            listt.append(entry.split("\\")[-1])
        for entry in glob.iglob(folder_path + '/**/*.bmp', recursive=True):
            print(entry)
            prediction = self.predict(entry, model, flat=flat, reshape=reshape)
            print("Prediction: " + listt[prediction.clone().detach().cpu().numpy()[0]])
            print("Real value: " + entry.split("\\")[-2])
            if str(prediction.clone().detach().cpu().numpy()[0]) == str(listt.index(entry.split("\\")[-2])):
                num_correct += 1
            else:
                num_incorrect += 1

        return (num_correct, num_incorrect)

    def predict(self, image_path, model=None, flat=False, reshape=True):
        img = io.imread(image_path)
        if model == None:
            model = self.model
        transform = model.get_val_transforms()
        if reshape:
            img = transform(img).reshape(1, list(transform(img).shape)[0], list(transform(img).shape)[1], list(transform(img).shape)[2])
        else:
            img = transform(img)
        return run_data_to_model((img, torch.tensor([0])), model.device, model.model, model.get_criterion(), model.get_optimizer(), train=False, get_prediction=True, flat_input=flat)
