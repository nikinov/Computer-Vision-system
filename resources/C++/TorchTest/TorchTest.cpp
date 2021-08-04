// TorchTest.cpp : Defines the entry point for the application.
//


#include <torch/script.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <chrono>
#include <iostream>
#include <memory>

using namespace std::chrono;

at::Tensor GetPrediction2(const char* modelPath, unsigned char byteArray[], int imwidth, int imhight);
unsigned char* matToBytes(cv::Mat image);
int main()
{
    std::cout << "yes" << std::endl;
    cv::Mat image = cv::imread("C:/!SAMPLES!/1716-5082/Sorted!/5/20210709-142903_X0Y0_R-scan1_b.bmp", 0);
    if (image.empty())
    {
        std::cout << "this image is empty" << std::endl;
    }
    else
    {
        std::cout << GetPrediction2("C:/Temp/models/jit_model.pt", matToBytes(image), image.size().width, image.size().height) << std::endl;

        std::cout << "showing image " << std::endl;
    }    
}

unsigned char* matToBytes(cv::Mat image)
{
    unsigned char* v_char = image.data;
    return v_char;
}

at::Tensor GetPrediction2(const char* modelPath, unsigned char imageData[], int imWidth, int imHeight)
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

    at::Tensor max_ind = at::argmax(output);

    std::cout << "class_id: " << max_ind.item<int>() << std::endl;
    std::cout << "Time take for forward pass: " << duration.count() << " ms" << std::endl;
    return output;
}

at::Tensor GetPrediction(const char* modelPath, unsigned char imageData[], int imwidth, int imheight)
{

    // Configuration
    int input_image_size = 28;

    // Deserialize the ScriptModule from a file using torch::jit::load().
    torch::jit::script::Module module;
    try {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        std::cout << "Loading model, path = " << "C:/Temp/models/model_num_naked.pt" << "\n";
        module = torch::jit::load("C:/Temp/models/model_num_naked.pt");
    }
    catch (const c10::Error& e) {
        std::cerr << "error loading the model\n";
    }

    int ptr = 0;

    //unsigned char* imageDataPtr = (unsigned char*)&imageData;
    cv::Mat img(imheight, imwidth, CV_8UC1, imageData);

    // Resize
    cv::Mat resized_image;
    cv::resize(img, resized_image, cv::Size(input_image_size, input_image_size));
    resized_image.convertTo(resized_image, CV_32F);
;
    // k_means clustering
    int k = 2;
    int attempts = 10;
    std::vector<int> labels;
    cv::Mat1f colors;
    const unsigned int singleLineSize = resized_image.rows * resized_image.cols;
    cv::Mat reshaped_image = resized_image.reshape(1, singleLineSize);
    cv::kmeans(reshaped_image, k, labels, cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 10, 1.), attempts, cv::KMEANS_PP_CENTERS, colors);

    // apply contrast changes
    for (unsigned int i = 0; i < singleLineSize; i++) { 
        reshaped_image.at<float>(i) = labels[i] * 255/(k-1);
    }

    //resized_image = reshaped_image.reshape(1, resized_image.rows);
    cv::Mat img_float;
    reshaped_image = reshaped_image + cv::Scalar(-127.5);
    reshaped_image.convertTo(img_float, CV_32F, 2.0 / 255);

    // turn into a tensor
    at::Tensor img_tensor = torch::from_blob(img_float.data, { 1, 784 });

    // Create a vector of inputs.
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(img_tensor.to(at::kCUDA));

    // Execute the model and turn its output into a tensor.
    at::Tensor output;
    auto duration = duration_cast<milliseconds>(std::chrono::high_resolution_clock::now() - std::chrono::high_resolution_clock::now());
    
    auto start = std::chrono::high_resolution_clock::now();
    output = module.forward(inputs).toTensor().to(at::kCUDA);
    std::cout << output << std::endl;
    auto end = std::chrono::high_resolution_clock::now();
    duration = duration_cast<milliseconds>(end - start);

    //at::Tensor max_ind = at::argmax(output);

    float out[10];
    for (unsigned int i = 0; i < 10; i++) {
        std::stringstream tmp;
        tmp << output[0][i].data();
        std::string tmp_str = tmp.str();
        out[i] = std::stof(tmp.str());
    }

    //std::cout << "class_id: " << max_ind.item<int>() << std::endl;
    std::cout << "Time take for forward pass: " << duration.count() << " ms" << std::endl;
    for (unsigned int i = 0; i < 10; i++) {
        std::cout << out[i] << std::endl;
    }
    
    return output;
}

float FlyFly()
{
    return 3;
}
