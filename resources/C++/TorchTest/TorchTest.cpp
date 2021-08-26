// TorchTest.cpp : Defines the entry point for the application.
//


#include <torch/script.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <chrono>
#include <iostream>
#include <memory>

using namespace std::chrono;

class custom_img {
    public:
        unsigned char* data;
        int width;
        int height;
};

void DLL_GetPrediction(unsigned char* imageData, int imHeight, int imWidth, float* out, int sizOut, bool resnet);
unsigned char* matToBytes(cv::Mat image);
void DLL_InitModel(const char* modelPath);
custom_img ReadBMP(char* filename);

// model
torch::jit::script::Module module;



int main()
{
    std::cout << "yes" << std::endl;
    //custom_img my_test_image = ReadBMP("C:/!SAMPLES!/1716-5082/Assets5082-samples-all/1/20210709-142856_X0Y0_R-scan1_b.bmp");
    cv::Mat my_test_image = cv::imread("C:/!SAMPLES!/1716-5082/Assets5082-samples-all/1/20210709-142856_X0Y0_R-scan1_b.bmp");
    if (my_test_image.empty())
    {
        std::cout << "this image is empty" << std::endl;
    }
    else
    {
        DLL_InitModel("C:/Users/Ryzen7-EXT/Documents/Github/WickonHightech/resources/python/new_vision_system/models/my_other_number_model.pt");
        float ouuut[11];
        DLL_GetPrediction(my_test_image.data, my_test_image.size().width, my_test_image.size().height, ouuut, 11, true);
        std::cout << ouuut << std::endl;

        std::cout << "showing image " << std::endl;
    }    
}

unsigned char* matToBytes(cv::Mat image)
{
    unsigned char* v_char = image.data;
    return v_char;
}

void DLL_InitModel(const char* modelPath) {
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
    int ptr = 0;

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
    resized_image.convertTo(img_float, CV_32F, 1.0/255.0);
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

custom_img ReadBMP(char* filename)
{
    int i;
    FILE* f = fopen(filename, "rb");

    if (f == NULL)
        throw "Argument Exception";

    unsigned char info[54];
    fread(info, sizeof(unsigned char), 54, f); // read the 54-byte header

    // extract image height and width from header
    int width = *(int*)&info[18];
    int height = *(int*)&info[22];

    std::cout << std::endl;
    std::cout << "  Name: " << filename << std::endl;
    std::cout << " Width: " << width << std::endl;
    std::cout << "Height: " << height << std::endl;

    int row_padded = (width * 3 + 3) & (~3);
    unsigned char* data = new unsigned char[row_padded];
    unsigned char tmp;

    for (int i = 0; i < height; i++)
    {
        fread(data, sizeof(unsigned char), row_padded, f);
        for (int j = 0; j < width * 3; j += 3)
        {
            // Convert (B, G, R) to (R, G, B)
            tmp = data[j];
            data[j] = data[j + 2];
            data[j + 2] = tmp;

            std::cout << "R: " << (int)data[j] << " G: " << (int)data[j + 1] << " B: " << (int)data[j + 2] << std::endl;
        }
    }

    fclose(f);
    custom_img out;
    out.data = data;
    out.height = height;
    out.width = width;
    return out;
}
