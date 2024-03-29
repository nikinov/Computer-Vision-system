// dllmain.cpp : Defines the entry point for the DLL application.
//#include <cstring>
#include <Windows.h>
#include <torch/script.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>

#include <chrono>
#include <iostream>
#include <memory>

using namespace std::chrono;
#undef min
#undef max

//#include "../Predictor/headers/Predictor.h"
#include "../headers/dllmain.h"


BOOL APIENTRY DllMain(HMODULE hModule,
    DWORD  ul_reason_for_call,
    LPVOID lpReserved
)
{
    switch (ul_reason_for_call)
    {
    case DLL_PROCESS_ATTACH:
    case DLL_THREAD_ATTACH:
    case DLL_THREAD_DETACH:
    case DLL_PROCESS_DETACH:
        break;
    }
    return TRUE;
}

torch::jit::script::Module module;


int DLL_test()
{
    return 0;
}

void DLL_InitModel(const char* modelPath) {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    try {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        std::cout << "Loading model, path = " << modelPath << "\n";
        module = torch::jit::load(modelPath);
    }
    catch (const c10::Error& e) {
        std::cerr << "error loading the model\n";
    }
}

void DLL_GetPrediction(unsigned char* imageData, int imWidth, int imHeight, float* out, int sizeOut, bool resnet) {
    // Configuration
    int input_image_size = 28;
    if (resnet)
        input_image_size = 224;

    //unsigned char* imageDataPtr = (unsigned char*)&imageData;
    cv::Mat img;
    if (resnet)
        img = cv::Mat(imHeight, imWidth, CV_8UC3, imageData);
    else
        img = cv::Mat(imHeight, imWidth, CV_8UC1, imageData);

    // Preprocess image (resize, put on GPU)
    cv::Mat resized_image;
    //cv::cvtColor(img, resized_image, cv::COLOR_RGB2GRAY);
    cv::resize(img, resized_image, cv::Size(input_image_size, input_image_size));
    cv::imwrite("C:/Temp/testing_img.png", resized_image);
    cv::Mat img_float;
    resized_image.convertTo(img_float, CV_32F, 1.0 / 255.0);
    at::Tensor img_tensor;
    if (!resnet) {
        img_tensor = torch::from_blob(img_float.data, { 1, input_image_size * input_image_size });
        img_tensor[0] = img_tensor[0].sub(0.5).div(0.5);
    }
    else {
        img_tensor = torch::from_blob(img_float.data, { 1, input_image_size, input_image_size, 3 });
        img_tensor = img_tensor.permute({ 0, 3, 1, 2 });
        img_tensor[0][0] = img_tensor[0][0].sub(0.5).div(0.5);
        img_tensor[0][1] = img_tensor[0][1].sub(0.5).div(0.5);
        img_tensor[0][2] = img_tensor[0][2].sub(0.5).div(0.5);
    }
    auto img_var = torch::autograd::make_variable(img_tensor, false);
    // Create a vector of inputs.
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(img_var.to(at::kCUDA));

    // Execute the model and turn its output into a tensor.
    at::Tensor output;
    auto duration = duration_cast<milliseconds>(std::chrono::high_resolution_clock::now() - std::chrono::high_resolution_clock::now());
    auto start = std::chrono::high_resolution_clock::now();
    std::cout << inputs.size() << std::endl;
    output = module.forward(inputs).toTensor().to(at::kCUDA);
    auto end = std::chrono::high_resolution_clock::now();
    duration = duration_cast<milliseconds>(end - start);
    std::cout << output << std::endl;
    // get values from tensor
    for (unsigned int i = 0; i < sizeOut; i++) {
        std::stringstream tmp;
        tmp << output[0][i].data();
        std::string tmp_str = tmp.str();
        out[i] = std::stof(tmp.str());
    }

    std::cout << "Time take for forward pass: " << duration.count() << " ms" << std::endl;
    for (unsigned int i = 0; i < sizeOut; i++) {
        std::cout << out[i] << std::endl;
    }
}


/*
int DLL_GetPrediction(
    const char* modelPath,
    unsigned char* byteArray,
    float* buffer,                      // output char array with version info message
    int allocSizOfBuffer                        // size of externally allocated array
)
{
    auto output = GetPrediction(modelPath, byteArray);

    std::memcpy(buffer, output.data<float>(), output.size(0));


    return 0;
}
*/