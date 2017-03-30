//
// Created by Mingkai Chen on 2017-03-30.
//

#include "proposal/proposal.hpp"

#ifdef LOGORNN_OBJECTPROPOSAL_HPP

namespace lrnn
{

std::vector<BOX> propose_objs (const cv::Mat& src,
	edge_params eparams,
	size_t min_size,
	size_t min_prop)
{
	// edge detect
	cv::Mat edges;
	lrnn::canny_thresh(src, edges, eparams);

	// watershed
	cv::Mat markers = src;
	size_t compCount = lrnn::watershed(src, edges, markers, min_size);

	// model the regions by adjacency graph
	cv::Mat adj_mat;
	lrnn::getAdjacencyMatrix(markers, adj_mat, compCount);

	// cache information on the regions
	lrnn::region_manager manager(src, markers, compCount);

	// perform hierarchy grouping
	lrnn::h_grouping(adj_mat,
		[&manager, &src](int i, int j) -> double
		{
			const lrnn::region_manager::region_info& infoI =
				manager.region_collect(i);

			const lrnn::region_manager::region_info& infoJ =
				manager.region_collect(j);

			double npixels = src.rows * src.cols;

			double colorscore = 0;
			for (int k = 0; k < infoI.color.nbins; k++)
			{
				colorscore += std::min(infoI.color.bin[k], infoJ.color.bin[k]);
			}

			double texturescore = 0;
			for (int k = 0; k < infoI.texture.nbins; k++)
			{
				texturescore += std::min(infoI.texture.bin[k], infoJ.texture.bin[k]);
			}

			double size_score = 1 - (double) (infoI.npixels + infoJ.npixels) / npixels;

			int loi = std::min(infoI.ul.first, infoJ.ul.first);
			int loj = std::min(infoI.ul.second, infoJ.ul.second);
			int hii = std::max(infoI.lr.first, infoJ.lr.first);
			int hij = std::max(infoI.lr.second, infoJ.lr.second);
			int di = hii - loi;
			int dj = hij - loj;
			double fill_score = 1 - ((di * dj) - infoI.npixels + infoJ.npixels) / npixels;

			// optional todo: add weights?
			return colorscore + texturescore + size_score + fill_score;
		},
		[&manager](int i, int j) -> int
		{
			return manager.region_merge(i, j);
		});

	std::vector<double> scores;
	std::default_random_engine generator;
	std::uniform_real_distribution<double> distribution(0.0,1.0);
	double rank = 1;
	for (auto rit = manager.hierarchy.rbegin(), ret = manager.hierarchy.rend();
		rit != ret; rit++)
	{
		double score = distribution(generator) * rank;
		scores.push_back(score);
		rank++;
	}

	std::vector<size_t> indices(scores.size());
	for (size_t i=0; i < scores.size(); i++) indices[i] = i;
	std::sort(indices.begin(), indices.end(),
		[&](size_t x, size_t y) -> bool { return scores[x] < scores[y]; });

	std::vector<BOX> boxes;
	for (size_t i = 0; i < std::min(indices.size(), min_prop); i++)
	{
		int phantomid = manager.hierarchy[indices[i]];
		const lrnn::region_manager::region_info& info = manager.region_collect(phantomid);
		std::vector<int> subs = info.subregions;
		// box super region encompassed by subs
		cv::Point tl = {info.ul.second, info.ul.first};
		cv::Point br = {info.lr.second, info.lr.first};
		boxes.push_back({tl, br});
	}
	return boxes;
}

}

#endif