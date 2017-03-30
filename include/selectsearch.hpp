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

	region_manager (const cv::Mat& src, const cv::Mat& marker, size_t nMarks) :
		src_(src), marker_(marker), nMarks(nMarks) {}

	~region_manager (void)
	{
		for (auto it : cache_)
		{
			delete it.second;
		}
	}

	/// collect information on region identified in marker from src
	const region_info& region_collect (int region)
	{
		auto it = cache_.find(region);
		if (cache_.end() != it)
		{
			return *(it->second);
		}
		int mini = marker_.rows;
		int minj = marker_.cols;
		int maxi = 0;
		int maxj = 0;
		int nchannels = src_.channels();
		region_info* info = new region_info(nchannels);
		for (int i = 0; i < marker_.rows; i++)
		{
			for (int j = 0; j < marker_.cols; j++)
			{
				int index = marker_.at<int>(i, j);
				if (region == index)
				{
					// color info
					for (int k = 0; k < nchannels; k++)
					{
						int c = src_.at<int>(i, j);
						int bucket = c * 25 / 255;
						info->color.bin[bucket + 25 * k]++;
					}
					// texture info
					// todo: collect texture info
					// size info
					info->npixels++;
					// corner info
					maxi = i > maxi ? i : maxi;
					mini = i < mini ? i : mini;
					maxj = j > maxj ? j : maxj;
					minj = j < minj ? j : minj;
				}
			}
		}
		info->ul = {mini, minj};
		info->lr = {maxi, maxj};
		cache_[region] = info;
		return *info;
	}

	/// create a phantom region in cache and catalog region hierarchy
	int region_merge (int regioni, int regionj)
	{
		int nchannels = src_.channels();
		// construct a virtual region in cache
		region_info* info = cache_[++nMarks] = new region_info(nchannels);
		hierarchy.push_back(nMarks);

		const region_info& infoI = region_collect(regioni);
		const region_info& infoJ = region_collect(regionj);

		if (infoI.subregions.empty())
		{
			info->subregions.push_back(regioni);
		}
		else
		{
			info->subregions.insert(info->subregions.end(),
				infoI.subregions.begin(), infoI.subregions.end());
		}

		if (infoJ.subregions.empty())
		{
			info->subregions.push_back(regionj);
		}
		else
		{
			info->subregions.insert(info->subregions.end(),
				infoJ.subregions.begin(), infoJ.subregions.end());
		}

		// merge corner info
		int loi = std::min(infoI.ul.first, infoJ.ul.first);
		int loj = std::min(infoI.ul.second, infoJ.ul.second);
		int hii = std::max(infoI.lr.first, infoJ.lr.first);
		int hij = std::max(infoI.lr.second, infoJ.lr.second);
		info->ul = {loi, loj};
		info->lr = {hii, hij};
		// merge size info
		size_t tpixels = info->npixels = infoI.npixels + infoJ.npixels;

		// merge color info
		for (int k = 0; k < info->color.nbins; k++)
		{
			info->color.bin[k] = (
				infoI.npixels * infoI.color.bin[k] +
				infoJ.npixels * infoJ.color.bin[k]) / tpixels;
		}
		return nMarks;
	}

	std::vector<int> get_subregions (int region)
	{
		return cache_[region]->subregions;
	}

	std::vector<int> hierarchy;

private:
	std::unordered_map<int, region_info*> cache_;

	size_t nMarks;
	const cv::Mat& src_;
	const cv::Mat& marker_;
};

}


#endif //LOGORNN_SELECTSEARCH_HPP
