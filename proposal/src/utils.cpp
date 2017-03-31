//
// Created by Mingkai Chen on 2017-03-27.
//

#include "utils.hpp"

#ifdef LOGORNN_UTILS_HPP

namespace lrnn
{

void blackenBG (cv::Mat& I)
{
	for (int x = 0; x < I.rows; x++)
	{
		for (int y = 0; y < I.cols; y++)
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

}

#endif
