//
// Created by Mingkai Chen on 2017-03-27.
//

#include <iostream>

#include "proposal.hpp"

using namespace cv;

Mat src;
RNG rng(12345);

lrnn::edge_params eparams;
const int kernel_size = 3;
const int min_size = 100;
const char* window_name = "Edge Map";
std::vector<double> weights;

int edge = 0;
const int max_edge = 30;

void segment (int, void*)
{
	eparams.sigma = 0.25 * edge * sqrt(2);
	Mat dest = src.clone();
	std::vector<lrnn::BOX> bounds = lrnn::propose_objs(src, eparams, min_size, 10, weights);
	if (bounds.size() == 0)
	{
		bounds.push_back({Point(0, 0), Point(src.cols, src.rows)});
	}
	for (lrnn::BOX b : bounds)
	{
		cv::Scalar color = cv::Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
		rectangle(dest, b.first, b.second, color, 2, 8, 0 );
	}

	imshow(window_name, dest);
}

int main (int argc, char** argv )
{
	if (argc < 2)
	{
		std::cout << "Usage: optest <image/path>" << std::endl;
		return -1;
	}
	double colorw = 1;
	double texturew = 1;
	double sizew = 1;
	double fillw = 1;
	if (argc >= 6)
	{
		colorw = atof(argv[2]);
		texturew = atof(argv[3]);
		sizew = atof(argv[4]);
		fillw = atof(argv[5]);
	}
	weights = {colorw, texturew, sizew, fillw};

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
	namedWindow(window_name, CV_WINDOW_AUTOSIZE);
	// Create a Trackbar for user to enter threshold
	createTrackbar( "Sigma ( *sqrt(2) ):", window_name, &edge, max_edge, segment );
	// Show the image
	segment(0, 0);
	// Wait until user exit program by pressing a key
	waitKey(0);

	return 0;
}