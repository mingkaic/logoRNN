//
// Created by Mingkai Chen on 2017-03-27.
//

#include "selectsearch.hpp"

#ifdef LOGORNN_SELECTSEARCH_HPP

namespace lrnn
{

void getAdjacencyMatrix (const cv::Mat& markers, cv::Mat& res, size_t compCount)
{
	int ncomp = compCount;
	// Create a KxK matrix and initialize to 0
	res = cv::Mat(compCount, compCount, CV_32S, cv::Scalar(0));

	// Scan the labeled image
	for (int i = 0; i < markers.rows; i++)
	{
		int left = -1;
		int right = -1;
		for (int j = 0; j < markers.cols; j++)
		{
			// Get the label of the current pixel and the ones of its neighbors
			int index = markers.at<int>(i, j);
			if (index <= 0 || index > compCount)
			{
				continue;
			}
			if (left <= 0 || right <= 0)
			{
				left = index;
				right = index;
				continue;
			}
			if (right != index)
			{
				res.at<int>(left-1, right-1) = 1;
				res.at<int>(right-1, left-1) = 1;

				left = right;
				right = index;
			}
		}
	}

	for (int j = 0; j < markers.cols; j++)
	{
		int up = -1;
		int down = -1;
		for (int i = 0; i < markers.rows; i++)
		{
			// Get the label of the current pixel and the ones of its neighbors
			int index = markers.at<int>(i, j);
			if (index <= 0 || index > compCount)
			{
				continue;
			}
			if (up <= 0 || down <= 0)
			{
				up = index;
				down = index;
				continue;
			}
			if (down != index)
			{
				res.at<int>(up-1, down-1) = 1;
				res.at<int>(down-1, up-1) = 1;

				up = down;
				down = index;
			}
		}
	}
}

struct similarity
{
	similarity (double score, int regioni, int regionj) :
		score(score), regioni(regioni), regionj(regionj) {}
	double score;
	int regioni;
	int regionj;
};

void h_grouping (const cv::Mat& adjMat,
	std::function<double(int,int)> compare,
	std::function<int(int,int)> merge)
{
	std::function<bool(similarity&,similarity&)> cmp =
		[](similarity& left, similarity& right)
		{
			return left.score < right.score;
		};
	// store all neighbors of each region
	std::unordered_map<int,std::vector<int> > adjs;
	// max-min priority queue of each region pair and their similarity score
	std::priority_queue<similarity, std::vector<similarity>,
		std::function<bool(similarity&,similarity&)>> S(cmp);
	for (size_t i = 1; i < adjMat.rows; i++)
	{
		for (size_t j = 0; j < i; j++)
		{
			if (adjMat.at<int>(i, j))
			{
				double score = compare(i+1, j+1);
				similarity sij(score, i+1, j+1);
				S.push(sij);
				adjs[i+1].push_back(j+1);
			}
		}
	}

	// merge regions
	while (false == S.empty())
	{
		const similarity& s = S.top();
		S.pop();
		auto iit = adjs.find(s.regioni);
		auto jit = adjs.find(s.regionj);
		if (adjs.end() == iit ||
			adjs.end() == jit)
		{
			continue;
		}
		int ri = merge(s.regioni, s.regionj);
		auto& avec = adjs[ri] = iit->second;
		auto& jvec = jit->second;
		avec.insert(avec.end(), jvec.begin(), jvec.end());
		for (int neighbor : avec)
		{
			double score = compare(ri, neighbor);
			similarity sij(score, ri, neighbor);
			S.push(sij);
		}
		adjs.erase(iit);
		adjs.erase(jit);
	}
}

region_manager::region_manager (const cv::Mat& src, const cv::Mat& marker, size_t nMarks) :
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
		grad_.push_back(cv::Mat::zeros(cv::Size(src.cols, src.rows), cv::DataType<double>::type));
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

region_manager::~region_manager (void)
{
	for (auto it : cache_)
	{
		delete it.second;
	}
}

const region_manager::region_info& region_manager::region_collect (int region)
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
	// normalize bins
	for (int k = 0; k < info->texture.nbins; k++)
	{
		info->texture.bin[k] /= info->npixels;
	}
	for (int k = 0; k < info->color.nbins; k++)
	{
		info->color.bin[k] /= info->npixels;
	}
	info->ul = {mini, minj};
	info->lr = {maxi, maxj};
	cache_[region] = info;
	return *info;
}

int region_manager::region_merge (int regioni, int regionj)
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

}

#endif