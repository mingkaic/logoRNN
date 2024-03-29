//
// Created by Mingkai Chen on 2017-04-02.
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
	// Load an image
	cv::Mat src = cv::imread(fname);

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
	size_t k = 3;
	if (argc == 7)
	{
		k = atoi(argv[6]);
	}
	std::vector<double> weights = {colorw, texturew, sizew, fillw};

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

	std::vector<lrnn::BOX> bounds = lrnn::propose_objs(src, eparams, min_size, k, weights);
	if (bounds.size() == 0)
	{
		bounds.push_back({cv::Point(0, 0), cv::Point(src.cols, src.rows)});
	}
	for (lrnn::BOX b : bounds)
	{
		std::cout << fname << " " << b.first.x << " " << b.first.y << " " << b.second.x << " " << b.second.y << std::endl;
	}

	return 0;
}