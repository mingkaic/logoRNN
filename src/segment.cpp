//
// Created by Mingkai Chen on 2017-03-27.
//

#include "segment.hpp"

#ifdef LOGORNN_SEGMENT_HPP

namespace lrnn
{

std::pair<double,double> athres (const cv::Mat& in)
{
	cv::Mat src_gray, mthres_g;
	cv::cvtColor(in, src_gray, CV_BGR2GRAY);

	double high_threshold = cv::threshold(src_gray, mthres_g, 0, 255,
										  CV_THRESH_BINARY | CV_THRESH_OTSU);
	double low_threshold = 0.1 * high_threshold;
	return {low_threshold, high_threshold};
}

void canny_thresh (const cv::Mat& in, cv::Mat& out,
	const edge_params& eparams)
{
	double lo_thres = eparams.lo_thres;
	double hi_thres = eparams.hi_thres;
	if (lo_thres < 0 || hi_thres < 0)
	{
		std::pair<double,double> lohi = athres(in);
		lo_thres = lohi.first;
		hi_thres = lohi.second;
	}

	cv::Mat src_gray, canny_in;
	cv::cvtColor(in, src_gray, CV_BGR2GRAY);

	cv::GaussianBlur(src_gray, canny_in, cv::Size(3,3), eparams.sigma);
	cv::Canny(canny_in, out, lo_thres, hi_thres, eparams.kernel_size);
}

size_t watershed (const cv::Mat& in, const cv::Mat& edge_in,
	cv::Mat& out, size_t min_size)
{
	// Find contours
	std::vector<std::vector<cv::Point> > contours;
	std::vector<cv::Vec4i> hierarchy;
	findContours(edge_in, contours, hierarchy,
		CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);

	// watershed
	out = cv::Mat(edge_in.size(), CV_32S);
	out = cv::Scalar::all(0);

	size_t compCount = 0;
	for (int idx = 0; idx >= 0;
		idx = hierarchy[idx][0], compCount++ )
	{
		if (fabs(cv::contourArea(contours[compCount])) < min_size)
		{
			continue;
		}
		drawContours(out, contours, idx, cv::Scalar::all(compCount+1), 1, 8, hierarchy, INT_MAX);
	}
	watershed(in, out);
	return compCount;
}


void color_label (const cv::Mat& in, const cv::Mat& comp_in,
	cv::Mat& out, size_t n_components)
{
	if (n_components == 0)
	{
		return;
	}
	// generate colors
	std::vector<cv::Vec3b> color_palette;
	for (int i = 0; i < n_components; i++)
	{
		int b = cv::theRNG().uniform(0, 255);
		int g = cv::theRNG().uniform(0, 255);
		int r = cv::theRNG().uniform(0, 255);
		color_palette.push_back(cv::Vec3b((uchar)b, (uchar)g, (uchar)r));
	}

	// paint the watershed image
	out = cv::Mat(comp_in.size(), CV_8UC3);
	for (int i = 0; i < comp_in.rows; i++)
	{
		for (int j = 0; j < comp_in.cols; j++)
		{
			int index = comp_in.at<int>(i,j);
			if (index == -1)
			{
				out.at<cv::Vec3b>(i,j) = cv::Vec3b(255,255,255);
			}
			else if (index <= 0 || index > n_components)
			{
				out.at<cv::Vec3b>(i,j) = cv::Vec3b(0,0,0);
			}
			else
			{
				out.at<cv::Vec3b>(i,j) = color_palette[index - 1];
			}
		}
	}

	// translucently paint source image onto watershed image
	cv::Mat src_gray;
	cv::Mat temp;
	cv::cvtColor(in, temp, cv::COLOR_BGR2GRAY);
	cv::cvtColor(temp, src_gray, cv::COLOR_GRAY2BGR);
	out = out*0.5 + src_gray*0.5;
}

}

#endif