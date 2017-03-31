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
void h_grouping (const cv::Mat& adjMat,
	std::function<double(int,int)> compare,
	std::function<int(int,int)> merge);

class region_manager
{
public:
	struct region_info
	{
		region_info (int nchannels) :
			color(25 * nchannels), texture(80 * nchannels) {}

		histo color;
		histo texture;
		size_t npixels = 0;
		coord ul = {0, 0}; // upper left corner
		coord lr = {0, 0}; // lower right corner

		std::vector<int> subregions;
	};

	region_manager (const cv::Mat& src, const cv::Mat& marker, size_t nMarks);

	~region_manager (void);

	/// collect information on region identified in marker from src
	const region_info& region_collect (int region);

	/// create a phantom region in cache and catalog region hierarchy
	int region_merge (int regioni, int regionj);

	std::vector<int> hierarchy;

private:
	std::unordered_map<int, region_info*> cache_;

	size_t nMarks;
	std::vector<cv::Mat> grad_;
	const cv::Mat& src_;
	const cv::Mat& marker_;
};

}


#endif //LOGORNN_SELECTSEARCH_HPP
