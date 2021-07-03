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
- make sure to install [pillow](https://pypi.org/project/Pillow/), [matplotlib](https://pypi.org/project/matplotlib/) and [torchsummary](https://pypi.org/project/torch-summary/)
- !! there are 2 links on the torch website, make sure to choose the debug version unless you need relese !!

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
:param save_config: if we wanna save the data config into txt, default=False
:param use_config: if we wanna use the data config from txt, default=False
"""
tr = train(dataset_path, out_path)
```
- prepare the and the data model for training
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

#### prep

For prep run the build_sln.bat file in resources\C++\PredictorDll, then go to the build folder and open up the solution and build it. It will generate a debug folder with the dll file that you can use when integrating with C#.

#### code

For integrating learning in C# call the following function. You can find an example in [source](https://github.com/nikinov/WickonHightech/tree/Torch/scr/Detector)
```C#

/// <summary>
/// Function interacts with a python training script and makes a model.pt file, all params are optional
/// </summary>
/// <param name="pythonPath">python path</param>
/// <param name="scriptPath">the pathe to the cli python intervafe script</param>
/// <param name="dataPath">path to your data or assets</param>
/// <param name="outPath">the path where the model.pt file will be outputed</param>
/// <param name="epoch">how many times will the training script go throught the data</param>
/// <param name="save_config">if we want to save data config into a txt file</param>
/// <param name="use_config">if we want to use a txt file with a data config</param>

using ModelMaker;

class Program
{
    static void Main(string[] args)
    {
        Maker.MakeModel();
        Console.ReadKey();
    }
}
```
For getting a prediction out of a model you can call the following function
```C#
/// <summary>
/// get a prediction in a form of an int
/// </summary>
/// <param name="modelPath"></param> path to the model
/// <param name="imageData"></param> image data in the form of a byte array
/// <param name="imHight"></param> image hight
/// <param name="imWidth"></param> image width
/// <returns>int representing the label</returns>

Predictor.GetPrediction();

```
