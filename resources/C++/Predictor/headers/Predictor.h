// Predictor.h : Include file for standard system include files,
// or project specific include files.

#pragma once

#include "C:/Users/Ryzen7-EXT/Documents/Github/WickonHightech/resources/C++/Predictor/lib/libtorch/include/torch/script.h"
//#include <torch/script.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <chrono>
#include <iostream>
#include <memory>

using namespace std::chrono;

int test() { return 3; }

int GetPrediction(const char* modelPath, unsigned char imageData[], int imHight, int imWidth)
{
    // Configuration
    int input_image_size = 224;
    int batch_size = 8;

    // Deserialize the ScriptModule from a file using torch::jit::load().
    torch::jit::script::Module module;
    try {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        module = torch::jit::load(modelPath);
    }
    catch (const c10::Error& e) {
        std::cerr << "error loading the model\n";
    }

    int ptr = 0;

    unsigned char* imageDataPtr = (unsigned char*)&imageData;
    cv::Mat img(imHight, imWidth, CV_8UC3, imageDataPtr);

    // Preprocess image (resize, put on GPU)
    cv::Mat resized_image;
    cv::cvtColor(img, resized_image, cv::COLOR_RGB2BGR);
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
//at::Tensor GetPrediction(const char* modelPath, unsigned char* byteArray);
/*
#include <iostream>
#include <torch/torch.h>

struct NetImpl : torch::nn::Module {
	NetImpl(int fc1_dims, int fc2_dims) :fc1(fc1_dims, fc1_dims), fc2(fc2_dims, fc2_dims), out(fc2_dims, 1) {
		register_module("fc1", fc1);
		register_module("fc2", fc2);
		register_module("out", out);
	}

	torch::Tensor forward(torch::Tensor x) {
		x = torch::relu(fc1(x));
		x = torch::relu(fc2(x));
		x = out(x);
		return x;
	}

	torch::nn::Linear fc1{ nullptr }, fc2{ nullptr }, out{ nullptr };
};

TORCH_MODULE(Net);
*/