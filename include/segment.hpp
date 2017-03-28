//
// Created by Mingkai Chen on 2017-03-27.
//

#include <vector>

#include "utils.hpp"

#ifndef LOGORNN_SEGMENT_HPP
#define LOGORNN_SEGMENT_HPP

namespace lrnn
{

// segment with edges using watershed
// ignore regions of pixels less than min_size
// return the number of segments
size_t segment (cv::Mat& out, const cv::Mat& in,
	size_t min_size,
	size_t kernel_size,
	int low_threshold,
	double ratio);

}

#endif //LOGORNN_SEGMENT_HPP
