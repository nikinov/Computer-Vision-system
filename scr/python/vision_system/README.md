# VisionAISystem
**Version 2.0.0**

---
📚 This is a quick guide on how to gert started with the VisionAISystem

## Before you start

---
To get started make sure you have cuda 11. Then install all the requirements with pip:

```shell
cd python/vision_system
pip install requirements.txt
```

## Prepare Data
___
To prepare the data create a folder with sub folders of classes with images in them that you want to train.

The data should have the following directory structure:

```
Images
├── cat
│   ├── cat0.png
│   ├── cat1.png
│   ├── cat2.png
│   └── cat3.png
└── dog
    ├── dog0.png
    ├── dog1.png
    ├── dog2.png
    └── dog3.png
```

## training and prediction
___
### For beginers
1. imports

Let's create a new python file inside the VisionAISystem directory in `python/vision_system` and then let's open up the file.

First let's define our imports to get us started:
```python
from train import AI
from utils import primary
from networks.pt_efficient_net import PtEfficientNet
```
___
2. train

Next let's make some variables and start the training process:

```python
# define my parameters
data_dir = "../../Assets"
model_output_path = "models/"
model_name = "my_favourite_model"
save_type = "jit_trace"

# define train object
tr = AI(model=PtEfficientNet())

# prepare for the training
tr.prep(dataset_path=data_dir, model_output_path=model_output_path)

# train our model
model = tr.train()

# save the model
primary.save_model(model, my_type="jit_trace", output_path=model_output_path, model_name=model_name)
```
___
3. load and test

After we have our trained model we can test it on a different dataset to see how it's performing
```python
# define my parameters
data_dir = "../../TestAssets"
model_path = "models/my_favourite_model.pt"
model_type = "jit"

# load my model
model = primary.load_model(model_path, my_type=model_type)

# define my AI object with a model the type of efficient_net
ai = AI(model=pt_efficient_net.PtEfficientNet(model=model))

# test our model a folder adn get the number of correct and incorrect images
num_correct, num_incorrect = ai.predict_folder(folder_path=data_dir)
print("number of correctly predicted images: " + num_correct)
print("number of incorrectly predicted images: " + num_incorrect)
```
___
4. predict

Now that we have a working and fully tested model we can finally put it to production and predict on our new images
```python
# my image path
image_path = "my_image.bmp"

# returns predicted label for the image
predicted_label = ai.predict(image_path)
print(predicted_label)
```
We can also utilise your bran new created model in our [C# module]()
___
### advanced

1. AI object and custom models

When you create the AI object you can define a custom model or an existing model of your choice 
all the models are located in the `networks` folder.

if you'd like to create your own model you could create it in `custom_networks.py` inside the 
`networks` or you can create a new script for it else ware.

To integrate your model with the `AI()` object you should create a new python script with a new 
class derived from ModelBase with the following functions that you can override as follows:
___
import your custom network with ModelBase
```python
from .model_base import ModelBase
from .custom_networks import MyCustomNN
```
___
Define your class.
```python
class MyCustomNetwork(ModelBase):
```

The following function should create and return the model input size. It could be fixed though it  
can also determine your input size Dynamically as shown in `pt_efficient_net.py` where it's determined 
by the optional parameter `average_input_size`, that is equal to the average size of an image in 
the dataset.
```python
    def make_input_size(self, average_input_size=None):
        my_input_size = 224
        return my_input_size
```

In this function you should fit all the preprocessing for the model to get it ready for training.
```python
    def model_prep(self):
        class_size = self.get_output_size()
        my_input_size = self.input_size
```

**optional! - the following is default**.

Here you can  do any preprocessing that needs to be done before saving the model.
```python
    def save_prep(self):
        pass
```

**optional! - the following is default**

Define your loss function.
```python
    def get_criterion(self):
        return nn.CrossEntropyLoss()
```

**optional! - the following is default**

Return an optimizer
```python
    def get_optimizer(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
```

**optional! - the following is default** 

return your transforms without any data augmentation
```python
    def get_val_transforms(self):
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((self.input_size, self.input_size)),
                transforms.Normalize((0.5,), (0.5,))
            ])
```
___

2. AI train and parameters

```python
from train import AI
from networks.pt_efficient_net import PtEfficientNet 
from data_preproessing.transforms import MyTransforms
from utils import primary


# define your parameters
model_name = "my_cool_model"
data_path  ="../../Assets"
model_output_path = "models/"
learning_rate = 0.001
batch_size = 10
generate_images_per_image = 40
train_transforms = MyTransforms.get_train_transforms()
save_type = None
epochs = 10
flat = False

# define your AI
ai = AI(
    model=PtEfficientNet( # your ModelBase derived model
        None, # pretrained existing model file otherwise a new one will be created
        model_name=model_name # model name stored in the ModelBase object
    ))                  

# prepare for training
ai.prep(
    dataset_path=data_path, # the path to your data
    model_output_path=model_output_path, # the path where the model file will be created
    learning_rate=learning_rate, # the rate at which the model is learning the higher the learning rate the bigger steps it takes at each epoch
    batch_size=batch_size, # the number of images should be dividable by the number of images or the number of generated images
    generate_images_per_image=generate_images_per_image, # number of images generated per every existing image
    train_trans=train_transforms, # transfroms for training including data augmentation for generating images
    visualisation=True # this will show you the preprocessed images if true
)

# train the model
model = ai.train(
    save_type=save_type, # you can specify if you want to save the model after training by replacing None with "pickle" for pickle "jit_trace" for trace and "jit_script" for script
    epochs=epochs, # number of iterations over the data
    flat=flat # if you want to turn the data into a one dimensional tensor before feeding into the model
)
```
___
3. test and predict

```python
# params
image_path = "my_nice_image.bmp"
test_folder = "../../TestAssets"
model_path = "models/my_cool_model.pt"
model_type = "jit"
reshape = True
flat = False

model = primary.load_model(
    model_path, # specify model path
    model_type # specify the type it could be ether pickle for pickle file or jit for pt files
)
ai = AI(
    model=PtEfficientNet( # your ModelBase derived model
        model, # pretrained existing model
    ) # model
)

# predict a folder of images
correct, incorrect = ai.predict_folder(
    folder_path=test_folder, #path to the folder you wanna test 
    model=None, # if you don't specify a model at AI you can specify it here
    flat=flat, # if you want to turn the data into a one dimensional tensor before feeding into the model
    reshape=reshape # if you need to reshape the input tensor from (shape) to (1, shape)
)

# predict a single image
prediction_label = ai.predict(
    image_path, #image path, otherwise the parameters are the same as in predict_folder()
)
```
___
Thanks for checking out my cool AI. Niki :)