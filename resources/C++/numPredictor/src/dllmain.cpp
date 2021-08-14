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

float *DLL_GetPrediction(unsigned char* imageData, int imHeight, int imWidth){
    // Configuration
    int input_image_size = 28;
    int ptr = 0;

    //unsigned char* imageDataPtr = (unsigned char*)&imageData;
    cv::Mat img(imHeight, imWidth, CV_8UC1, imageData);
    // Preprocess image (resize, put on GPU)
    cv::Mat resized_image;
    //cv::cvtColor(img, resized_image, cv::COLOR_RGB2GRAY);
    cv::resize(img, resized_image, cv::Size(input_image_size, input_image_size));
    cv::imwrite("C:/Temp/testing_img.png", resized_image);
    cv::Mat img_float;
    resized_image.convertTo(img_float, CV_32F, 1.0 / 255);

    auto img_tensor = torch::from_blob(img_float.data, { 1, input_image_size * input_image_size });
    img_tensor[0] = img_tensor[0].sub(0.5).div(0.5);
    auto img_var = torch::autograd::make_variable(img_tensor, false);



    // Create a vector of inputs.
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(img_var.to(at::kCUDA));

    // Execute the model and turn its output into a tensor.
    at::Tensor output;
    auto duration = duration_cast<milliseconds>(std::chrono::high_resolution_clock::now() - std::chrono::high_resolution_clock::now());

    auto start = std::chrono::high_resolution_clock::now();
    output = module.forward(inputs).toTensor().to(at::kCUDA);
    auto end = std::chrono::high_resolution_clock::now();
    duration = duration_cast<milliseconds>(end - start);

    // get values from tensor
    float out[11];
    for (unsigned int i = 0; i < 11; i++) {
        std::stringstream tmp;
        tmp << output[0][i].data();
        std::string tmp_str = tmp.str();
        out[i] = std::stof(tmp.str());
    }

    std::cout << "Time take for forward pass: " << duration.count() << " ms" << std::endl;
    for (unsigned int i = 0; i < 11; i++) {
        std::cout << out[i] << std::endl;
    }
    return out;
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