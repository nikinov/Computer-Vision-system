# TorchAISystem
**Version 1.0.0**

This is a an image classifier that uses pytorch in Python and torchscript in C++ to train and make a prediction on a inseption model.

check out other branches for much more!!!

## Before you start

---
To get started make sure you have cuda 11 with a [compatible GPU](https://developer.nvidia.com/cuda-gpus). Then install all the requirements with pip:

```shell
pip install requirements.txt
```
after the pip installation follow these steps for setting op the rest

Setup
-----
### Installation & Compilation HOW-TO


CUDA
----
install: cuda_11.3.1_465.89_win10.exe


CudNN
-----
copy CudNN files: cudnn-11.2-windows-x64-v8.1.1.33.zip
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vx.x\


LIBTORCH
--------
https://pytorch.org/ 
- Stable(1.9.0) + Windows + LibTorch + C++/Java + CUDA11.1 -> Download DEBUG(!) version
- unpack to C:\libtorch (libtorch-win-shared-with-deps-debug-1.9.0+cu111.zip)


Build MODEL
-----------
- run LibTorch\resources\python\example.py


Compile PredictorDLL
--------------------
LibTorch\resources\C++\PredictorDll\
- run build_sln.bat
- rebuild build\Predictor_Dll.sln in VS2019


Compile and test TorchTest (optional)
-------------------------------------
LibTorch\resources\C++\TorchTest\
- run build_sln.bat
- rebuild and run build\TorchTest.sln in VS2019


Compile and test Detector C#
----------------------------
LibTorch\scr\Detector\
- rebuild and run Detector.sln


MW, 2.7.2021

### How to local tutorials
follow [this](https://github.com/nikinov/WickonHightech/tree/CleanTorch/src/python/vision_system) tutorial for AI vision system

or go [here](https://github.com/nikinov/WickonHightech/tree/CleanTorch/src/python/yolo) for yolo

tutorials for C++ and C# coming soon

### how to online tutorials
you can follow [this tutorial](https://learnopencv.com/image-classification-using-transfer-learning-in-pytorch/) for the python script
as well as [this tutorial](https://www.youtube.com/watch?v=Dk88zv1KYMI) for deploymant in C++
