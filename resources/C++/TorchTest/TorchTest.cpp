// TorchTest.cpp : Defines the entry point for the application.
//


#include <torch/script.h> 
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <chrono>
#include <iostream>
#include <memory>

using namespace std::chrono;

at::Tensor GetPrediction(const char* modelPath, unsigned char byteArray[], int imwidth, int imhight);
unsigned char* matToBytes(cv::Mat image);
int main()
{
    cv::Mat image = cv::imread("C:/!SAMPLES!/1716-5082/digit2/20210709-142856_X0Y0_R-scan0.bmp", 0);
    if (image.empty())
    {
        std::cout << "this image is empty" << std::endl;
    }
    else
    {
        std::cout << GetPrediction("C:/Temp/models/num_model.pt", matToBytes(image), image.size().width, image.size().height) << std::endl;
        std::cout << "showing image" << std::endl;
    }    
}

unsigned char* matToBytes(cv::Mat image)
{
    unsigned char* v_char = image.data;
    return v_char;
}

at::Tensor GetPrediction(const char* modelPath, unsigned char imageData[], int imwidth, int imheight)
{

    // Configuration
    int input_image_size = 28;

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
    cv::Mat img(imheight, imwidth, CV_8UC1, imageData);
    // Preprocess image (resize, put on GPU)
    cv::Mat resized_image;
    cv::resize(img, resized_image, cv::Size(input_image_size, input_image_size));
    cv::imshow("test", resized_image);
    cv::waitKey();

    cv::Mat img_float;
    resized_image.convertTo(img_float, CV_32F, 1.0 / 255);

    auto img_tensor = torch::from_blob(img_float.data, { 1, input_image_size, input_image_size });
    img_tensor = img_tensor.permute({ 0, 1, 2 });
    img_tensor[0][0] = img_tensor[0][0].sub(0.5).div(0.5);
    img_tensor[0][1] = img_tensor[0][1].sub(0.5).div(0.5);
    img_tensor[0][2] = img_tensor[0][2].sub(0.5).div(0.5);
    auto img_var = torch::autograd::make_variable(img_tensor, false);

    // Create a vector of inputs.
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(img_var.to(at::kCPU));
    std::cout << inputs << std::endl;
    // Execute the model and turn its output into a tensor.
    at::Tensor output;
    auto duration = duration_cast<milliseconds>(std::chrono::high_resolution_clock::now() - std::chrono::high_resolution_clock::now());

    auto start = std::chrono::high_resolution_clock::now();
    output = module.forward(inputs).toTensor().to(at::kCPU);
    auto end = std::chrono::high_resolution_clock::now();
    duration = duration_cast<milliseconds>(end - start);

    at::Tensor max_ind = at::argmax(output);

    std::cout << "class_id: " << max_ind.item<int>() << std::endl;
    std::cout << "Time take for forward pass: " << duration.count() << " ms" << std::endl;
    return output;
}

