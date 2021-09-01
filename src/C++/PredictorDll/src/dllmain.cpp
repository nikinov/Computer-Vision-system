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

int DLL_test()
{
    return 0;
}
int DLL_GetPrediction(const char* modelPath, unsigned char* imageData, int imHight, int imWidth)
{
    // Configuration
    int input_image_size = 224;
    int batch_size = 8;

    // Deserialize the ScriptModule from a file using torch::jit::load().
    torch::jit::script::Module module;
    try {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        std::cout << "Loading model, path = " << modelPath << "\n";
        module = torch::jit::load(modelPath);
    }
    catch (const c10::Error& e) {
        std::cerr << "error loading the model\n";
    }

    int ptr = 0;

    //unsigned char* imageDataPtr = (unsigned char*)&imageData;
    cv::Mat img(imHight, imWidth, CV_8UC4, imageData);
    cv::imshow("image",img);
    cv::waitKey(0);
    // Preprocess image (resize, put on GPU)
    cv::Mat resized_image;
    try {
        cv::cvtColor(img, resized_image, cv::COLOR_RGB2BGR);
    }
    catch (const c10::Error& e)
    {
        std::cerr << "your byte array is incorrect, try redoing your byte array. Hint: openCV doesn't like bmp files but it can work with png or jpg\n";
    }
    cv::resize(resized_image, resized_image, cv::Size(input_image_size, input_image_size));

    cv::Mat img_float;
    resized_image.convertTo(img_float, CV_32F, 1.0 / 255);

    auto img_tensor = torch::from_blob(img_float.data, { 1, input_image_size, input_image_size, 3 });
    img_tensor = img_tensor.permute({ 0, 3, 1, 2 });
    img_tensor[0][0] = img_tensor[0][0].sub(0.485).div(0.229);
    img_tensor[0][1] = img_tensor[0][1].sub(0.456).div(0.224);
    img_tensor[0][2] = img_tensor[0][2].sub(0.406).div(0.225);
    auto img_var = torch::autograd::make_variable(img_tensor, false);

    // Create a vector of inputs.
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(img_var.to(at::kCUDA));

    // Execute the model and turn its output into a tensor.
    at::Tensor output;
    auto duration = duration_cast<milliseconds>(std::chrono::high_resolution_clock::now() - std::chrono::high_resolution_clock::now());
    for (int i = 0; i < 10; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        output = module.forward(inputs).toTensor().to(at::kCPU);
        auto end = std::chrono::high_resolution_clock::now();
        duration = duration_cast<milliseconds>(end - start);
    }

    at::Tensor max_ind = at::argmax(output);

    std::cout << "class_id: " << max_ind.item<int>() << std::endl;
    std::cout << "Time take for forward pass: " << duration.count() << " ms" << std::endl;
    return max_ind.item<int>();
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