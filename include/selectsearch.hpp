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
		src_(src), marker_(marker), nMarks(nMarks)
	{
		std::function<void(const cv::Mat&,cv::Mat&)> calc_grad =
		[](const cv::Mat& in, cv::Mat& out)
			{
				int scale = 1;
				int delta = 0;
				int ddepth = CV_16S;
				cv::Mat grad_x, grad_y;
				cv::Mat abs_grad_x, abs_grad_y;

				cv::Sobel(in, grad_x, ddepth, 1, 0, 3, scale, delta, cv::BORDER_DEFAULT);
				cv::Sobel(in, grad_y, ddepth, 0, 1, 3, scale, delta, cv::BORDER_DEFAULT);

				for (int i = 0; i < in.rows; i++)
				{
					for (int j = 0; j < in.cols; j++)
					{
						double gx = grad_x.at<int>(i, j);
						double gy = grad_y.at<int>(i, j);
						out.at<double>(i, j) = atan2(gy, gx);
					}
				}
			};

		size_t nchannels = src.channels();
		for (size_t i = 0; i < nchannels; i++)
		{
			grad_.push_back(cv::Mat::zeros(cv::Size(src.rows, src.cols), cv::DataType<double>::type));
		}

		// get gradient
		if (nchannels > 1)
		{
			std::vector<cv::Mat> msplit;
			// split the channels
			cv::split(src, msplit);
			// apply kernel to each channel
			for (int i = 0; i < 3; i++)
			{
				calc_grad(msplit[i], grad_[i]);
			}
		}
		else
		{
			calc_grad(src, grad_[0]);
		}
	}

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
		std::function<void(int,int)> colorhist = [&nchannels, info, this](int i, int j)
		{
			cv::Vec3b c = src_.at<cv::Vec3b>(i, j);
			for (int k = 0; k < nchannels; k++)
			{
				int bucket = c[k] * 25 / 255;
				info->color.bin[bucket + 25 * k]++;
			}
		};
		std::function<void(int,int)> texthist = [&nchannels, info, this](int i, int j)
		{
			cv::Vec3b c = src_.at<cv::Vec3b>(i, j);
			for (int k = 0; k < nchannels; k++)
			{
				int gidx = grad_[k].at<double>(i, j) * 4 / M_PI + 4;
				int coloridx = c[k] * 10 / 255;
				info->texture.bin[gidx + 8 * coloridx + 80 * k]++;
			}
		};
		if (nchannels == 1)
		{
			colorhist = [info, this](int i, int j)
			{
				int c = src_.at<char>(i, j);
				int bucket = c * 25 / 255;
				info->color.bin[bucket]++;
			};
			texthist = [info, this](int i, int j)
			{
				int gidx = grad_[0].at<double>(i, j) * 4 / M_PI + 4;
				int coloridx = src_.at<char>(i, j);
				info->texture.bin[gidx + 8 * coloridx]++;
			};
		}
		for (int i = 0; i < marker_.rows; i++)
		{
			for (int j = 0; j < marker_.cols; j++)
			{
				int index = marker_.at<int>(i, j);
				if (region == index)
				{
					// color info
					colorhist(i, j);
					// texture info
					texthist(i, j);
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
		// merge texture info
		for (int k = 0; k < info->texture.nbins; k++)
		{
			info->texture.bin[k] = (
				infoI.npixels * infoI.texture.bin[k] +
				infoJ.npixels * infoJ.texture.bin[k]) / tpixels;
		}
		// merge color info
		for (int k = 0; k < info->color.nbins; k++)
		{
			info->color.bin[k] = (
				infoI.npixels * infoI.color.bin[k] +
				infoJ.npixels * infoJ.color.bin[k]) / tpixels;
		}
		return nMarks;
	}

	std::pair<coord,coord> get_box (int region)
	{
		return {cache_[region]->ul, cache_[region]->lr};
	}

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
