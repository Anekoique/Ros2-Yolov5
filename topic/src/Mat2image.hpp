#ifndef MAT2IMG_H
#define MAT2IMG_H
#include "opencv2/opencv.hpp"
#include <vector>
#include "image_base64.hpp"

class Mat2Img 
{
public:
	static int MatToByteArray(const cv::Mat mat, std::vector<unsigned char>& buff)
	{
		if (mat.empty()) {
			return 0;
		}
		std::vector<int> param = std::vector<int>(2);
		param[0] = cv::IMWRITE_JPEG_QUALITY;
		param[1] = 95; // default(95) 0-100
		cv::imencode(".jpg", mat, buff, param);
		return 0;
	}
	
	static int JPEGToMat(cv::Mat &matImage, std::vector<unsigned char> buff)
	{
		if (buff.empty()) {
			return -1;
		}
		if ((buff[0] == 0xFF && buff[1] == 0xD8))
		{
			matImage = cv::imdecode(buff, cv::IMREAD_COLOR);
		}
		else
		{
			buff.insert(buff.begin(), 0xFF);
			buff.insert(buff.begin()+1, 0xD8);
			matImage = cv::imdecode(buff, cv::IMREAD_COLOR);
		}
		return 0;
	}

	static int Mat2Base64(cv::Mat &image, std::string &base64)
	{
		std::vector<unsigned char> cut_jpg;
		cut_jpg.clear();
		Mat2Img::MatToByteArray(image, cut_jpg);

		base64 = ImageBase64::encode(cut_jpg.data(), cut_jpg.size());
		return 0;
	}
	static int Base2Mat(std::string &base64, cv::Mat &image)
	{
		
		int outLen = 0;
		std::string str1 = ImageBase64::decode(base64.c_str(), base64.size(), outLen);

		std::vector<uchar> baseJpgVec;
		baseJpgVec.assign(&str1.data()[0], &str1.data()[outLen]);

		int ret = Mat2Img::JPEGToMat(image, baseJpgVec);
		if (ret != 0)
		{
			return ret;
		}

		return 0;
	}
};
#endif
