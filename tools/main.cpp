//
// Created by Mingkai Chen on 2017-03-27.
//

#include <random>
#include <iostream>

#include "segment.hpp"
#include "selectsearch.hpp"

using namespace cv;

Mat src;

lrnn::edge_params eparams;
const int kernel_size = 3;
const int min_size = 100;
const char* window_name = "Edge Map";

int edge = 0;
const int max_edge = 30;

void segment (int, void*)
{
	eparams.sigma = edge * sqrt(2);

	// edge detect
	cv::Mat edges;
	lrnn::canny_thresh(src, edges, eparams);

	// watershed
	Mat markers = src;
	size_t compCount = lrnn::watershed(src, edges, markers, min_size);

	// model the regions by adjacency graph
	Mat adj_mat;
	lrnn::getAdjacencyMatrix(markers, adj_mat, compCount);

	// cache information on the regions
	lrnn::region_manager manager(src, markers, compCount);

	// perform hierarchy grouping
	lrnn::h_grouping(adj_mat,
	[&manager](int i, int j) -> double
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

	vector<size_t> indices(scores.size());
	for (size_t i=0; i < scores.size(); i++) indices[i] = i;
	std::sort(indices.begin(), indices.end(),
	[&](size_t x, size_t y) -> bool { return scores[x] < scores[y]; });

	// take half
	for (size_t i = 0; i < indices.size(); i++)
	{
		int phantomid = manager.hierarchy[indices[i]];
		std::vector<int> subs = manager.get_subregions(phantomid);
		// todo: box super region encompassed by subs
	}

	// display watershed image
	Mat wshed;
	lrnn::color_label(src, markers, wshed, compCount);
	imshow(window_name, wshed);
}

int main (int argc, char** argv )
{
	if (argc < 2)
	{
		std::cout << "Usage: logoRNN <image/path>" << std::endl;
		return -1;
	}

	// Load an image
	src = imread( argv[1] );

	// Check if everything was fine
	if (!src.data)
	{
		std::cout << "Data not found at " << argv[1] << std::endl;
		return -1;
	}

	// adaptive threshold
	std::pair<double,double> lohi = lrnn::athres(src);
	eparams.kernel_size = kernel_size;
	eparams.lo_thres = lohi.first;
	eparams.hi_thres = lohi.second;

	// Create a window
	namedWindow(window_name, CV_WINDOW_AUTOSIZE);
	// Create a Trackbar for user to enter threshold
	createTrackbar( "Sigma ( *sqrt(2) ):", window_name, &edge, max_edge, segment );
	// Show the image
	segment(0, 0);
	// Wait until user exit program by pressing a key
	waitKey(0);

	return 0;
}