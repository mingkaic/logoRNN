//
// Created by Mingkai Chen on 2017-03-27.
//

#include "segment.hpp"

#ifdef LOGORNN_SEGMENT_HPP

namespace lrnn
{

size_t segment (cv::Mat& out, const cv::Mat& in,
	size_t min_size,
	size_t kernel_size,
	int low_threshold,
	double ratio)
{
	// edge detect
	cv::Mat edges;
	lrnn::canny_thresh(edges, in, kernel_size, low_threshold, ratio);

	// Find contours
	std::vector<std::vector<cv::Point> > contours;
	std::vector<cv::Vec4i> hierarchy;
	findContours(edges, contours, hierarchy,
		CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);

	// watershed
	out = cv::Mat(edges.size(), CV_32S);
	out = cv::Scalar::all(0);

	size_t compCount = 0;
	for(int idx = 0; idx >= 0;
		idx = hierarchy[idx][0], compCount++ )
	{
		if (fabs(cv::contourArea(contours[compCount])) < min_size)
		{
			continue;
		}
		drawContours(out, contours, idx, cv::Scalar::all(compCount+1), 1, 8, hierarchy, INT_MAX);
	}
	watershed(in, out);
	return compCount;
}

}

#endif