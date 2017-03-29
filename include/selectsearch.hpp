//
// Created by Mingkai Chen on 2017-03-27.
//

#include <queue>
#include <vector>
#include <unordered_set>
#include <unordered_map>

#include "utils.hpp"

#ifndef LOGORNN_SELECTSEARCH_HPP
#define LOGORNN_SELECTSEARCH_HPP

namespace lrnn
{

/// extract adjacency matrix from input markers obtained from opencv's watershed
void getAdjacencyMatrix (const cv::Mat& markers, cv::Mat& res, size_t compCount);

/// group regions by similarities
void h_grouping (const cv::Mat& adjMat, cv::Mat& grouped,
	std::function<double(int,int)> compare,
	std::function<int(int,int)> merge);

}


#endif //LOGORNN_SELECTSEARCH_HPP
