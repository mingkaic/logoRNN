//
// Created by Mingkai Chen on 2017-04-06.
//

#include <iostream>

#include "proposal.hpp"

int main (int argc, char** argv )
{
	if (argc < 2)
	{
		std::cerr << "Usage: opgen <image/path>" << std::endl;
		return -1;
	}

	std::string fname = argv[1];
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
	bool water_prop = false;
	if (argc == 7)
	{
		water_prop = true;
	}

	std::vector<double> weights = {colorw, texturew, sizew, fillw};
	// Load an image
	cv::Mat src = cv::imread(fname);

	// Check if everything was fine
	if (!src.data)
	{
		std::cerr << "Data not found at " << argv[1] << std::endl;
		return -1;
	}

	// adaptive threshold
	const int min_size = 100;
	std::pair<double,double> lohi = lrnn::athres(src);
	lrnn::edge_params eparams;
	eparams.lo_thres = lohi.first;
	eparams.hi_thres = lohi.second;
	eparams.kernel_size = 3;
	eparams.sigma = 0.75 * sqrt(2); // works for most images

	intermediate(src, eparams, min_size, 10, weights, water_prop);
	cv::waitKey(0);

	return 0;
}