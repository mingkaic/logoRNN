//
// Created by Mingkai Chen on 2017-03-30.
//

#include "egbis.h"

#include "proposal.hpp"

#ifdef LOGORNN_OBJECTPROPOSAL_HPP

namespace lrnn
{

static int mark_egdis (const cv::Mat& egdis, cv::Mat& out)
{
	out = cv::Mat::zeros(cv::Size(egdis.cols, egdis.rows), cv::DataType<int>::type);
	std::unordered_map<uint32_t,int> labelmap;
	int label = 1;
	for (int i = 0; i < egdis.rows; i++)
	{
		for (int j = 0; j < egdis.cols; j++)
		{
			cv::Vec3b col = egdis.at<cv::Vec3b>(i, j);
			uint32_t index = (col[0] << 16) + (col[1] << 8) + col[2];
			auto it = labelmap.find(index);
			if (it == labelmap.end())
			{
				labelmap.emplace(index, label);
				out.at<int>(i, j) = label;
				label++;
			}
			else
			{
				out.at<int>(i, j) = it->second;
			}
		}
	}
	return label;
}

static void paint (const cv::Mat& markers, cv::Mat& out, std::unordered_set<int>& hset)
{
	int b = cv::theRNG().uniform(0, 255);
	int g = cv::theRNG().uniform(0, 255);
	int r = cv::theRNG().uniform(0, 255);
	cv::Vec3b samecolor((uchar)b, (uchar)g, (uchar)r);
	for (int i = 0; i < markers.rows; i++)
	{
		for (int j = 0; j < markers.cols; j++)
		{
			int index = markers.at<int>(i,j);
			if (hset.find(index) != hset.end())
			{
				out.at<cv::Vec3b>(i, j) = samecolor;
			}
		}
	}
}

void intermediate (const cv::Mat& src,
   edge_params eparams,
   size_t min_size,
   size_t min_prop,
   std::vector<double> weights,
   bool water_prop)
{
	cv::Mat markers;
	size_t compCount;

	if (water_prop)
	{
		// use edge detect + watershed
		cv::Mat edges;
		lrnn::canny_thresh(src, edges, eparams);

		// watershed
		markers = src;
		compCount = lrnn::watershed(src, edges, markers, min_size);
	}
	else
	{
		// use efficient graph-based method
		int nccs;
		cv::Mat egdis_img = runEgbisOnMat(src, eparams.sigma, 500, 500, &nccs);

		compCount = mark_egdis(egdis_img, markers);
	}

	cv::Mat seg;
	color_label(src, markers, seg, compCount);

	imshow("segmentation", seg);

	// model the regions by adjacency graph
	cv::Mat adj_mat;
	lrnn::getAdjacencyMatrix(markers, adj_mat, compCount);

	// cache information on the regions
	lrnn::region_manager manager(src, markers, compCount);

	// perform hierarchy grouping
	lrnn::h_grouping(adj_mat,
		 [&manager, &src, &weights](int i, int j) -> double
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

			 if (weights.size() == 4)
			 {
				 return weights[0] * colorscore + weights[1] * texturescore + weights[2] * size_score + weights[3] * fill_score;
			 }
			 else
			 {
				 return colorscore + texturescore + size_score + fill_score;
			 }
		 },
		 [&manager](int i, int j) -> int
		 {
			 return manager.region_merge(i, j);
		 });

	size_t hierarchy_height = manager.hierarchy.size();

	std::stringstream ss;
	for (size_t i = 0; i < hierarchy_height; i++)
	{
		const region_manager::region_info& hinfo =
			manager.region_collect(manager.hierarchy[i]);
		std::vector<int> hsubs = hinfo.subregions;
		std::unordered_set<int> hset = std::unordered_set<int>(hsubs.begin(), hsubs.end());
		paint(markers, seg, hset);
		ss << "level " << i;
		imshow(ss.str(), seg);
		ss.str("");
	}
}

std::vector<BOX> propose_objs (const cv::Mat& src,
	edge_params eparams,
	size_t min_size,
	size_t min_prop,
	std::vector<double> weights)
{
#ifdef WATERSHED_PROP
	// use edge detect + watershed
	cv::Mat edges;
	lrnn::canny_thresh(src, edges, eparams);

	// watershed
	cv::Mat markers = src;
	size_t compCount = lrnn::watershed(src, edges, markers, min_size);
#else
	// use efficient graph-based method
	int nccs;
	cv::Mat egdis_img = runEgbisOnMat(src, eparams.sigma, 500, 500, &nccs);

	cv::Mat markers;
	size_t compCount = mark_egdis(egdis_img, markers);
#endif

	// model the regions by adjacency graph
	cv::Mat adj_mat;
	lrnn::getAdjacencyMatrix(markers, adj_mat, compCount);

	// cache information on the regions
	lrnn::region_manager manager(src, markers, compCount);

	// perform hierarchy grouping
	lrnn::h_grouping(adj_mat,
		[&manager, &src, &weights](int i, int j) -> double
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

			if (weights.size() == 4)
			{
				return weights[0] * colorscore + weights[1] * texturescore + weights[2] * size_score + weights[3] * fill_score;
			}
			else
			{
				return colorscore + texturescore + size_score + fill_score;
			}
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

	size_t nscores = scores.size();
	std::vector<size_t> indices(nscores);
	for (size_t i=0; i < scores.size(); i++) indices[i] = i;
	std::sort(indices.begin(), indices.end(),
		[&](size_t x, size_t y) -> bool { return scores[x] < scores[y]; });

	std::vector<BOX> boxes;
	for (size_t i = 0; i < min_prop && i < indices.size(); i++)
	{
		int phantomid = manager.hierarchy[nscores - indices[i] - 1];
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