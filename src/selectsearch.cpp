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
	auto cmp =
		[](similarity& left, similarity& right)
		{
			return left.score < right.score;
		};
	// store all neighbors of each region
	std::unordered_map<int,std::vector<int> > adjs;
	// max-min priority queue of each region pair and their similarity score
	std::priority_queue<similarity, std::vector<similarity>, decltype(cmp)> S(cmp);
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

}

#endif