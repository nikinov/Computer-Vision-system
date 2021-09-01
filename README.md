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

### Setup
-----
### Installation & Compilation HOW-TO


#### PYTORCH
-------
pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio===0.9.0 -f https://download.pytorch.org/whl/torch_stable.html


#### CUDA
----
install: cuda_11.3.1_465.89_win10.exe


#### CudNN
-----
copy CudNN files: cudnn-11.2-windows-x64-v8.1.1.33.zip
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vx.x\


#### LIBTORCH
--------
https://pytorch.org/ 
- Stable(1.9.0) + Windows + LibTorch + C++/Java + CUDA11.1 -> Download DEBUG(!) version
- unpack to C:\libtorch (libtorch-win-shared-with-deps-debug-1.9.0+cu111.zip)


#### Build MODEL
-----------
- run LibTorch\resources\python\example.py


#### Compile PredictorDLL
--------------------
LibTorch\resources\C++\PredictorDll\
- run build_sln.bat
- rebuild build\Predictor_Dll.sln in VS2019


#### Compile and test TorchTest (optional)
-------------------------------------
LibTorch\resources\C++\TorchTest\
- run build_sln.bat
- rebuild and run build\TorchTest.sln in VS2019


#### Compile and test Detector C#
----------------------------
LibTorch\scr\Detector\
- rebuild and run Detector.sln


MW, 2.7.2021

### How to Make

you can follow [this tutorial](https://learnopencv.com/image-classification-using-transfer-learning-in-pytorch/) for the python script
as well as [this tutorial](https://www.youtube.com/watch?v=Dk88zv1KYMI) for deploymant in C++

### pytorch training

follow [this](https://github.com/nikinov/WickonHightech/tree/CleanTorch/src/python/vision_system) tutorial for AI vision system

or go [here](https://github.com/nikinov/WickonHightech/tree/CleanTorch/src/python/yolo) for yolo
