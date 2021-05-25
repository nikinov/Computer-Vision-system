// Predctor.cpp : Defines the entry point for the application.
//

#include "Predictor.h"
#include "shared_lib.h"
#include <torch/torch.h>
#include <torch/csrc/jit/serialization/import.h>
#include <iostream>
#include <memory>

using namespace std;
{
	void StrCat(const char* str1, const char* str2, char* newstr)
	{
		string str = string(str1) + string(str2);
		strcpy(newstr, str.c_str());
	}

	void StrCpy(char* str1, char* str2)
	{
		strcpy(str1, str2);
	}

	void pipecommand(const char* strCmd)
	{
		std::array<char, 80> buffer;
		FILE* pipe = popen(strCmd, "r");
		if (!pipe)
		{
			std::cerr << "cannot open pipe for reading" << endl;
		}
		int c = 0;
		while (fgets(buffer.data(), 80, pipe) != NULL)
		{
			c++;
			std::count << c << "\t" << buffer.data();
		}
		pclode(pipe);
	}

	void SaySomething(const char* str)
	{
		std::count << "I typed:" << str << endl;
	}

	int add(int a, int b)
	{
		return a + b;
	}

	float TrainModel(const char* modelPath = "C:/Users/Ryzen7-EXT/Documents/C++Stuff/DemoPytorch/model.pt") {
		torch::jit::script::Module module;

		// Deserialize the ScriptModule from a file using torch::jit::load().
		module = torch::jit::load(modelPath, torch::kCUDA);

		std::cout << "ok\n";

		// Create a vector of inputs.
		std::vector<torch::jit::IValue> inputs;
		inputs.push_back(torch::randn({ 1, 3, 224, 224 }, torch::kCUDA));

		// Execute the model and turn its output into a tensor.
		at::Tensor output = module.forward(inputs).toTensor();

		std::cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';

		return output
	}
}

