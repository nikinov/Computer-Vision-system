# TorchAISystem
[Discord server](https://discord.gg/Qcme78MB)
**Version 1.0.0**

This is a an image classifier that uses pytorch in Python and torchscript in C++ to train and make a prediction on a inseption model.

## Tutorial

This is an example tutorial on setup, make and deploy a Pytorch model.

### Prerequisites

Make sure you have installed all of the following prerequisites on your development machine:

- make sure you have a [compatible GPU](https://developer.nvidia.com/cuda-gpus)
- [Python](https://www.python.org/), feel free to use [Anaconda](https://www.anaconda.com/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html), when going through the install process make sure to select add conda to path!!!
- [Pytorch](https://pytorch.org/) make sure to select the following options, though if you use conda select the conda option instead of pip:
<img width="613" alt="Screenshot 2021-05-14 at 1 36 25" src="https://user-images.githubusercontent.com/54107324/118200266-07bd5300-b455-11eb-81eb-3ddc7af0bddc.png">
- uninstall all NVIDIA drivers and files and then get [Cuda toolkit](https://developer.nvidia.com/accelerated-computing-toolkit) and [CuDNN](https://developer.nvidia.com/cudnn), make sure it's the same as the one mentioned in pytorch, you can also follow [this tutorial](https://www.youtube.com/watch?v=raBkhUoeOHs) for pytorch and cuda installation
- [Visual Studio 2019](https://visualstudio.microsoft.com/) with C++
- [LibTorch](https://pytorch.org/), for this process you can follow [this tutorial](https://www.youtube.com/watch?v=6eTVqYGIWx0)

### How to Make

you can follow [this tutorial](https://learnopencv.com/image-classification-using-transfer-learning-in-pytorch/) for the python script
as well as [this tutorial](https://www.youtube.com/watch?v=Dk88zv1KYMI) for deploymant in C++

### pytorch training

this is an example of how to use the train_c.c or train script
- import and initialisation 
```python
from train_c import train

dataset_path = "../assets"
out_path = "/models"

"""
initialise train
:param dataset_path: path for the assets, default="../../Assets"
:param model_output_path: model output path, default="../models"
"""
tr = train(dataset_path, out_path)
```
- prepare the and the data model for training
first make a directory structure simmilar to this
<img width="168" alt="Screenshot 2021-05-20 at 22 28 34" src="https://user-images.githubusercontent.com/54107324/119044622-b8c66f00-b9ba-11eb-91e4-de6c396ccf86.png">
a directory that countains a 3 subdirectories each containing a subdirectory for each class. Call tr.model_prep() for model prep and tr.train_and_validate() for training and validation

```python
"""
Prepare the model
:param resnet_type: type of resnet model, default=resnet152()
"""
tr.model_prep()

"""
Train and validate the model
:param model: the model to train
:param loss_criterion: the loss function
:param optimizer: the optimizer for the model
:param epochs: the number of iterations
:param show_results: plot data about the training process
:return: a trained model with the highest accuracy
"""

tr.train_and_validate(epochs=40)
```

- then you can make a prediction on a single image
```python
'''
Predict the class of a single test image
Parameters
    :param model: Model to test
    :param test_image_path: Test image path, default=trained resnet model
'''

tr.predict('testimage.png')
```

### integration

comming soon
