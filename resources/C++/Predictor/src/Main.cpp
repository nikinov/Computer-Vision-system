// Predictor.cpp : Defines the entry point for the application.
//

#include <iostream>
#include "../headers/Predictor.h"
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace torch;
using namespace std;
using namespace cv;

int main()
{
	string path = "C:/Users/Ryzen7-EXT/Documents/Github/WickonHightech/resources/C++/Predictor/Resources/test.png";
	Mat img = imread(path);
	imshow("Image", img);
	waitKey(0);
}
