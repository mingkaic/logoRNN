//
// Created by Mingkai Chen on 2017-03-27.
//

#include <iostream>

#include "segment.hpp"

using namespace cv;

Mat src;

lrnn::edge_params eparams;
const int kernel_size = 3;
const int min_size = 100;
const char* window_name = "Edge Map";

int edge = 0;
const int max_edge = 30;

void segment (int, void*)
{
	eparams.sigma = edge * sqrt(2);

	// edge detect
	cv::Mat edges;
	lrnn::canny_thresh(src, edges, eparams);

	// watershed
	Mat markers = src;
	size_t compCount = lrnn::watershed(src, markers, edges, min_size);

	// display watershed image
	Mat wshed;
	lrnn::color_label(src, wshed, markers, compCount);
	imshow(window_name, wshed);
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

	// adaptive threshold
	std::pair<double,double> lohi = lrnn::athres(src);
	eparams.kernel_size = kernel_size;
	eparams.lo_thres = lohi.first;
	eparams.hi_thres = lohi.second;

	// Create a window
	namedWindow( window_name, CV_WINDOW_AUTOSIZE );
	// Create a Trackbar for user to enter threshold
	createTrackbar( "Sigma ( *sqrt(2) ):", window_name, &edge, max_edge, segment );
	// Show the image
	segment(0, 0);
	// Wait until user exit program by pressing a key
	waitKey(0);

	return 0;
}