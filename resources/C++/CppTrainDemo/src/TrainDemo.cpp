
#include <torch/torch.h>
#include <torch/script.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <chrono>
#include <iostream>
#include <memory>
#include <filesystem>

namespace fs = std::filesystem;

int TrainModel();

int main()
{
	TrainModel();
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

	return temp;
}

std::tuple<float, float> evaluate_model(torch::jit::script::Module model,
	std::vector<data> preprocessed_data, torch::nn::NLLLoss criterion, int image_num)
{
	torch::NoGradGuard no_grad;
	model.eval();
	model.to(at::kCUDA);

	int64_t num_running_corrects = 0;
	int64_t num_samples = 0;
	float running_loss = 0;

	// Iterate the data loader to yield batches from the dataset.
	for (int i = image_num; i < image_num; i++) {
		data data_use = preprocessed_data.at(i);

		auto image = data_use.image;
		auto label = data_use.label;
		torch::Tensor outputs = model.forward(image).toTensor().to(at::kCUDA);
		torch::Tensor preds = std::get<1>(torch::max(outputs, 1));
		torch::Tensor lebs = torch::from_blob(label.data(), { (int64_t)label.size() }, at::TensorOptions().dtype(at::kShort)).clone();
		num_running_corrects += torch::sum(preds == lebs).item<int64_t>();

		torch::Tensor loss_tensor = criterion(outputs, lebs);
		float loss = loss_tensor.item<float>();
		num_samples += image.size();
		running_loss += loss * image.size();
	}

	float eval_accuracy =
		static_cast<float>(num_running_corrects) / num_samples;
	float eval_loss = running_loss / num_samples;

	return { eval_accuracy, eval_loss };
}

int TrainModel()
{
	// Configuration
	// std::vector<int> hidden_size = {128, 64};
	int hidden_size_0 = 128;
	int hidden_size_1 = 64;
	int input_size = 784;
	int input_image_size = 28;
	int k = 2;
	int epochs = 55;
	int image_num = 0;
	int class_num = 0;
	int data_generate_num = 64;
	const static char* files = "C:/!SAMPLES!/1716-5082/Sorted!";
	std::vector<std::string> label_names;


	// Get Num of Classes and num of images
	for (const auto& dirEntry : fs::recursive_directory_iterator(files)) {
		if (dirEntry.is_directory()) {
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
	torch::jit::script::Module model;
	try {
		// Deserialize the ScriptModule from a file using torch::jit::load().
		std::cout << "Loading model, path = " << "C:/Temp/models/model_num_naked.pt" << "\n";
		model = torch::jit::load("C:/Temp/models/model_num_naked.pt");
	}
	catch (const c10::Error& e) {
		std::cerr << "error loading the model\n";
	}
	/*torch::nn::Sequential model(
		torch::nn::Linear(input_size, hidden_size_0),
		torch::nn::ReLU(),
		torch::nn::Linear(hidden_size_0, hidden_size_1),
		torch::nn::ReLU(),
		torch::nn::Linear(hidden_size_1, class_num),
		torch::nn::LogSoftmax()
	);*/

	//torch::Device device("cpu");
	//if (torch::cuda::is_available())
	//{
	//	device = torch::Device("cuda:0");
	//}

	// iterate over all images preprocess then and add them to array
	int i = 0;
	for (const auto& dirEntry : fs::recursive_directory_iterator(files)) {
		if (fs::path(dirEntry).extension() == ".bmp") {
			std::vector<torch::jit::IValue> inputs;

			// Get image
			cv::Mat img = cv::imread(dirEntry.path().u8string(), 0);

			// Resize
			cv::resize(img, img, cv::Size(input_image_size, input_image_size));

			//for (cv::Mat img : dataAugmentationRotation(img, data_generate_num))
		//{
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
			inputs.push_back(img_tensor.to(at::kCUDA));

			//}

			// add everything to preprocessed images
			data preprocessed_data;
			preprocessed_data.image = inputs;
			//for (int j = 0; j < data_generate_num; j++) {
			preprocessed_data.label.push_back(getIndex(label_names, fs::path(dirEntry).parent_path().filename().u8string()));
			//}
			preprocessed_images.push_back(preprocessed_data);
			i++;
		}
	}
	
	// training and validation
	torch::nn::NLLLoss criterion{};
	std::vector<at::Tensor> parameters;
	for (const auto& parameter : model.parameters()) {
		parameters.push_back(parameter.data());
	}
	torch::optim::SGD optimizer{ parameters, torch::optim::SGDOptions(0.05).momentum(0.9) };

	for (int epoch = 0; epoch < epochs; epoch++) {
		// training loop
		model.train();
		model.to(at::kCUDA);
		float running_loss_train = 0;
		int64_t num_running_corrects = 0;
		int64_t num_samples = 0;
		for (int i = 0; i < image_num; i++) {

			data data_use = preprocessed_images.at(i);

			auto image = data_use.image;
			auto label = data_use.label;
			
			optimizer.zero_grad();
			torch::Tensor outputs = model.forward(image).toTensor().to(at::kCUDA);

			auto preds = std::get<1>(torch::max(outputs, 1)).to(at::kCUDA);
			torch::Tensor lebs = torch::full({ 1 }, label.at(0)).to(at::kCUDA);//torch::from_blob(label.data(), { (int64_t)label.size() }, at::TensorOptions().dtype(at::kLong)).clone().to(at::kCUDA);
			num_running_corrects += torch::sum(preds == lebs).item<int64_t>();
			auto options = torch::TensorOptions().dtype(torch::kLong);
			torch::Tensor loss = criterion(outputs, lebs).to(at::kCUDA);
			num_samples += image.size();
			
			loss.backward();

			optimizer.step();
			running_loss_train += loss.item().toFloat() * image.size();
		}
		float train_accuracy = static_cast<float>(num_running_corrects) / num_samples;
		float train_loss = running_loss_train / num_samples;

		//model.eval();

		//std::tuple<float, float> eval_result =
			//evaluate_model(model, preprocessed_images, criterion, image_num);
		std::cout << std::setprecision(6) << "Epoch: " << std::setfill('0')
			<< std::setw(3) << epoch << " Train Loss: " << train_loss
			<< " Train Acc: " << train_accuracy << std::endl;
			//<< " Eval Loss: " << std::get<1>(eval_result)
			//<< " Eval Acc: " << std::get<0>(eval_result) 

		/*
		// validation loop
		model->eval();
		model->to(device);
		torch::NoGradGuard no_grad;
		int64_t num_running_corrects = 0;
		int64_t num_samples = 0;
		float running_loss_valid = 0;
		for (int i = image_num - (int)image_num / 10; i < image_num; i++) {
			data data_use = preprocessed_images.at(i);

			auto image = data_use.image;
			auto label = data_use.label;
			torch::Tensor outputs = model->forward(image);
			torch::Tensor preds = std::get<1>(torch::max(outputs, 1));

		}
	}
	// save model
	std::string model_path = "C:/Temp/models/num_model2.pt";
	torch::serialize::OutputArchive output_archive;
	model->save(output_archive);
	output_archive.save_to(model_path);
*/
	}
	return 1;
}


