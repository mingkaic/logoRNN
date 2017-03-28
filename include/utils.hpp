//
// Created by Mingkai Chen on 2017-03-27.
//

#include <opencv2/opencv.hpp>

#ifndef LOGORNN_UTILS_HPP
#define LOGORNN_UTILS_HPP

namespace lrnn
{

/// change pure white to black
void blackenBG (cv::Mat& I);

/// canny edge operator
void canny_thresh (cv::Mat& out, const cv::Mat& in,
	size_t kernel_size, int low_threshold, double ratio);

}

#endif // LOGORNN_UTILS_HPP
