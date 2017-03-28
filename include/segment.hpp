//
// Created by Mingkai Chen on 2017-03-27.
//

#include <vector>

#include "utils.hpp"

#ifndef LOGORNN_SEGMENT_HPP
#define LOGORNN_SEGMENT_HPP

namespace lrnn
{

/// adaptive threshold using otsu + binary thresholding method
std::pair<double,double> athres (const cv::Mat& in);

/// canny edge operator
void canny_thresh (const cv::Mat& in, cv::Mat& out,
	size_t kernel_size, double sigma,
	double lo_thres, double hi_thres);

/// segment with edges using watershed
/// ignore regions of pixels less than min_size
/// return the number of segments
size_t watershed (const cv::Mat& in, cv::Mat& out,
	const cv::Mat& edge_in, size_t min_size);

/// label components in comp_in with colors
/// transluciently add regions to original source image in
/// through resulting image out
void color_label (const cv::Mat& in, cv::Mat& out,
	const cv::Mat& comp_in, size_t n_components);

}

#endif //LOGORNN_SEGMENT_HPP
