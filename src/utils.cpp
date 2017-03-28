//
// Created by Mingkai Chen on 2017-03-27.
//

#include "utils.hpp"

#ifdef LOGORNN_UTILS_HPP

namespace lrnn
{

void blackenBG (cv::Mat& I)
{
	for( int x = 0; x < I.rows; x++ )
	{
		for( int y = 0; y < I.cols; y++ )
		{
			if (I.at<cv::Vec3b>(x, y) == cv::Vec3b(255,255,255))
			{
				I.at<cv::Vec3b>(x, y)[0] = 0;
				I.at<cv::Vec3b>(x, y)[1] = 0;
				I.at<cv::Vec3b>(x, y)[2] = 0;
			}
		}
	}
}

void canny_thresh (cv::Mat& out, const cv::Mat& in,
	size_t kernel_size,
	int low_threshold,
	double ratio)
{
	cv::Mat src_gray;

	cv::cvtColor(in, src_gray, CV_BGR2GRAY );
	cv::GaussianBlur(src_gray, out, cv::Size(3,3), sqrt(2));
	cv::Canny(out, out, low_threshold, low_threshold*ratio, kernel_size);
}

}

#endif
