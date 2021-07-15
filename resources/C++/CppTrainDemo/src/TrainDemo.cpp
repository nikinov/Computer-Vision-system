#include <torch/script.h> 
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <chrono>
#include <iostream>
#include <memory>
#include <filesystem>

namespace fs = std::filesystem;

void TrainModel();

int main()
{
	//TrainModel();
}
class data {
	public:
		std::vector<torch::jit::IValue> image;
		std::vector<int> label;
};

int getIndex(std::vector<std::string> v, std::string K)
{
	auto it = find(v.begin(), v.end(), K);

	// If element was found
	if (it != v.end())
	{

		// calculating the index
		// of K
		int index = it - v.begin();
		return index;
	}
	else {
		// If the element is not
		// present in the vector
		throw "something wrong with the labels TrainDemo.cpp";
		return -1;
	}
}

std::vector<cv::Mat> dataAugmentationRotation(cv::Mat img, int num_of_images) {
	
	std::vector<cv::Mat> temp;
	cv::Point2f pc(img.cols/2., img.rows / 2.);
	for (int i = 0; i < num_of_images; i++) {
		cv::Mat r = cv::getRotationMatrix2D(pc, ((float)rand() / RAND_MAX*2)-1, 1.0);
		temp.push_back(r);
	}

	return;
}

void TrainModel()
{
	// Configuration
	// std::vector<int> hidden_size = {128, 64};
	int hidden_size_0 = 128;
	int hidden_size_1 = 64;
	int input_size = 784;
	int input_image_size = 28;
	int k = 2;
	int epochs = 15;
	int image_num = 0;
	int class_num = 0;
	const static char* files = "C:/!SAMPLES!/1716-5082/Sorted!";
	std::vector<std::string> label_names;
	

	// Get Num of Classes and num of images
	for (const auto& dirEntry : fs::recursive_directory_iterator(files)) {
		if (dirEntry.is_directory()){
			label_names.insert(label_names.begin(), fs::path(dirEntry).filename().u8string());
			class_num += 1;
		}
		else if (fs::path(dirEntry).extension() == ".bmp") {
			image_num += 1;
		}
	}
	const int image_num_in = image_num;
	std::vector<data> preprocessed_images;

	// Instantiate model
	auto model = torch::nn::Sequential(
		torch::nn::Linear(input_size, hidden_size_0),
		torch::nn::ReLU(),
		torch::nn::Linear(hidden_size_0, hidden_size_1),
		torch::nn::ReLU(),
		torch::nn::Linear(hidden_size_1, class_num),
		torch::nn::LogSoftmax()
	);

	torch::Device device("cpu");
	if (torch::cuda::is_available())
	{
		device = torch::Device("cuda:0");
	}

	// iterate over all images preprocess then and add them to array
	int i;
	for (const auto & dirEntry : fs::recursive_directory_iterator(files)) {
		if (fs::path(dirEntry).extension() == ".bmp") {
			std::vector<torch::jit::IValue> inputs;
			
			// Get image
			cv::Mat img = cv::imread(dirEntry.path().u8string(), 0);

			// Resize
			cv::resize(img, img, cv::Size(input_image_size, input_image_size));

			for (cv::Mat img : dataAugmentationRotation(img, 64))
			{
				img.convertTo(img, CV_32F);

				// Reashape image into color_space * number_of_pixles
				std::vector<int> labels;
				cv::Mat1f colors;
				const unsigned int singleLineSize = img.rows * img.cols;
				cv::Mat reshaped_image = img.reshape(1, singleLineSize);

				// Apply k_means
				int attempts = 10;
				cv::kmeans(reshaped_image, k, labels, cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 10, 1.), attempts, cv::KMEANS_PP_CENTERS, colors);
			
				// Apply contrast changes
				for (unsigned int i = 0; i < singleLineSize; i++) {
					reshaped_image.at<float>(i) = labels[i] * 255 / (k - 1);
				}

				// Rescale the image to fit the model
				cv::Mat img_float;
				reshaped_image = reshaped_image + cv::Scalar(-127.5);
				reshaped_image.convertTo(img_float, CV_32F, 2.0 / 255);

				// Convert into a tensor
				at::Tensor img_tensor = torch::from_blob(img_float.data, { 1, 784 });

				// Create a vector of inputs.
				inputs.push_back(img_tensor.to(device));

			}

			// add everything to preprocessed images
			data preprocessed_data;
			preprocessed_data.image = inputs;
			preprocessed_data.label.push_back(getIndex(label_names, fs::path(dirEntry).parent_path().filename().u8string()));
			preprocessed_images.push_back(preprocessed_data);
			i++;
		}
	}
	// training loop

	model->to(device);
	auto criterion = torch::nn::NLLLoss();

	auto optimizer = torch::optim::SGD(model->parameters(), torch::optim::SGDOptions(0.003).momentum(0.9));

	for (int epoch = 0; epoch < epochs; epoch++) {
		model->train();

		// training loop
		float runnning_loss = 0;
		for (int i = 0; i < image_num - (int)image_num/10; i++) {
			
			data data_use = preprocessed_images.at(i);

			auto image = data_use.image;
			auto label = data_use.label;

			optimizer.zero_grad();

			torch::Tensor output = model->forward(image);
			
			torch::Tensor loss = criterion(output, label);

			loss.backward();

			optimizer.step();

			runnning_loss += loss.item().toFloat();
		}

		// validation loop
		for (int i = image_num - (int)image_num / 10; i < image_num; i++) {

		}
	}

	// save model
}


