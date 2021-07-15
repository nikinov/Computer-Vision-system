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
    cv::Mat image = cv::imread("C:/!SAMPLES!/1716-5082/Sorted!/5/20210709-142903_X0Y0_R-scan1_b.bmp", 0);
    if (image.empty())
    {
        std::cout << "this image is empty" << std::endl;
    }
    else
    {
        //std::cout << GetPrediction("C:/Temp/models/num_model.pt", matToBytes(image), image.size().width, image.size().height) << std::endl;
        std::vector<int> temp[2];
        temp->push_back(2);
        temp->push_back(3);
        temp->push_back(4);
        temp->at(2) = 10;
        std::cout << "showing image " << temp->at(2) << "||" << temp->at(1) << std::endl;
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
        std::cout << "Loading model, path = " << "C:/Temp/models/num_model.pt" << "\n";
        module = torch::jit::load("C:/Temp/models/num_model.pt");
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
    std::cout << img_tensor  << std::endl;
    output = module.forward(inputs).toTensor().to(at::kCUDA);
    
    auto end = std::chrono::high_resolution_clock::now();
    duration = duration_cast<milliseconds>(end - start);

    //at::Tensor max_ind = at::argmax(output);

    float out[10];
    for (unsigned int i = 0; i < 9; i++) {
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
