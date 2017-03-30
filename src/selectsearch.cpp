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
	for (int i = 1; i < markers.rows-1; i++)
	{
		for (int j = 1; j < markers.cols-1; j++)
		{
			// Get the label of the current pixel and the ones of its neighbors
			int k 		= markers.at<int>(i, j);
			int kleft 	= markers.at<int>(i-1, j);
			int kright 	= markers.at<int>(i+1, j);
			int kup 	= markers.at<int>(i, j-1);
			int kdown 	= markers.at<int>(i, j+1);

			if (k <= 0 || k > compCount ||
				kleft <= 0 || kleft > compCount ||
				kright <= 0 || kright > compCount ||
				kup <= 0 || kup > compCount ||
				kdown <= 0 || kdown > compCount)
			{
				continue;
			}
			if (k != kleft)
			{
				res.at<int>(k-1, kleft-1) = 1;
				res.at<int>(kleft-1, k-1) = 1;
			}
			if (k != kright)
			{
				res.at<int>(k-1, kright-1) = 1;
				res.at<int>(kright-1, k-1) = 1;
			}
			if (k != kup)
			{
				res.at<int>(k-1, kup-1) = 1;
				res.at<int>(kup-1, k-1) = 1;
			}
			if (k != kdown)
			{
				res.at<int>(k-1, kdown-1) = 1;
				res.at<int>(kdown-1, k-1) = 1;
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
				double score = compare(i, j);
				similarity sij(score, i, j);
				S.push(sij);
				adjs[i].push_back(j);
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