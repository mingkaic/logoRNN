//
// Created by Mingkai Chen on 2017-03-30.
//

#include <random>
#include <algorithm>

#include "segment.hpp"
#include "selectsearch.hpp"

#ifndef LOGORNN_OBJECTPROPOSAL_HPP
#define LOGORNN_OBJECTPROPOSAL_HPP

namespace lrnn {

using BOX = std::pair<cv::Point,cv::Point>;

// eparams are a set of parameters for detecting edge (sigma, kernel size, etc), see segment.hpp
// min_size is the minimum number of pixels for each segment pre-selection search
// min_prop is the minimum number of proposals outputted.
std::vector<BOX> propose_objs (const cv::Mat& src,
	edge_params eparams,
	size_t min_size,
	size_t min_prop);

}

#endif //LOGORNN_OBJECTPROPOSAL_HPP
