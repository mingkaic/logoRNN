//
// Created by Mingkai Chen on 2017-03-27.
//

#include <cstring>
#include <opencv2/opencv.hpp>

#ifndef LOGORNN_UTILS_HPP
#define LOGORNN_UTILS_HPP

namespace lrnn
{

using coord = std::pair<int,int>;

struct histo
{
	histo (size_t nbins) : nbins(nbins)
	{
		bin = new double[nbins];
		std::memset(bin, 0, sizeof(double) * nbins);
	}

	~histo (void) { delete[] bin; }

	double* bin;
	size_t nbins;
};

/// change pure white to black
void blackenBG (cv::Mat& I);

}

#endif // LOGORNN_UTILS_HPP
