//
// Created by Mingkai Chen on 2017-03-27.
//

#include <iostream>

#include "segment.hpp"

using namespace cv;

Mat src, dst;
int lowThreshold;
int ratio = 3;
int kernel_size = 3;

const int min_size = 100;
const char* window_name = "Edge Map";

const int max_lowThreshold = 100;

void watershed (int, void*)
{
	Mat markers = src;
	size_t compCount = lrnn::segment(markers, src, min_size, kernel_size, lowThreshold, ratio);

	if( compCount == 0 )
		return;
	vector<Vec3b> colorTab;
	for(int i = 0; i < compCount; i++)
	{
		int b = theRNG().uniform(0, 255);
		int g = theRNG().uniform(0, 255);
		int r = theRNG().uniform(0, 255);
		colorTab.push_back(Vec3b((uchar)b, (uchar)g, (uchar)r));
	}

	Mat wshed(markers.size(), CV_8UC3);
	// paint the watershed image
	for(int i = 0; i < markers.rows; i++)
	{
		for(int j = 0; j < markers.cols; j++)
		{
			int index = markers.at<int>(i,j);
			if( index == -1 )
				wshed.at<Vec3b>(i,j) = Vec3b(255,255,255);
			else if( index <= 0 || index > compCount )
				wshed.at<Vec3b>(i,j) = Vec3b(0,0,0);
			else
				wshed.at<Vec3b>(i,j) = colorTab[index - 1];
		}
	}
	cv::Mat src_gray;
	cv::Mat temp;
	cvtColor(src, temp, COLOR_BGR2GRAY);
	cvtColor(temp, src_gray, COLOR_GRAY2BGR);
	wshed = wshed*0.5 + src_gray*0.5;
	imshow( window_name, wshed );
}


int main(int argc, char** argv )
{
	if (argc < 2)
	{
		std::cout << "Usage: logoRNN <image/path>" << std::endl;
		return -1;
	}

	// Load an image
	src = imread( argv[1] );

	// Check if everything was fine
	if (!src.data)
	{
		std::cout << "Data not found at " << argv[1] << std::endl;
		return -1;
	}
	// Create a window
	namedWindow( window_name, CV_WINDOW_AUTOSIZE );
	// Create a Trackbar for user to enter threshold
	createTrackbar( "Min Threshold:", window_name, &lowThreshold, max_lowThreshold, watershed );
	// Show the image
	watershed(0, 0);
	// Wait until user exit program by pressing a key
	waitKey(0);

	return 0;
}