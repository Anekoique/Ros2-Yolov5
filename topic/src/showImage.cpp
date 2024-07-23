#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "cv_bridge/cv_bridge.h"
#include "Mat2image.hpp"
#include <chrono>

using namespace std;
using namespace cv;

class showImage : public rclcpp::Node
{
public:
	showImage() : Node("show") {
		subscriber_ = this->create_subscription<sensor_msgs::msg::Image>(
				"image_raw",
				10,
				std::bind(&showImage::image_callback, this, std::placeholders::_1)
		);
	}
private:
	rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscriber_;
	VideoCapture cap;
	VideoWriter video;
	int isFirstFrame = 1;

	void image_callback(const sensor_msgs::msg::Image::SharedPtr msg) {
		RCLCPP_INFO(this->get_logger(), "Receiving video frames");
		Mat src;
		std::string base64 = std::string((msg->data).begin(), (msg->data).end());
		Mat2Img::Base2Mat(base64, src);

		namedWindow("camera", WINDOW_NORMAL);
		resizeWindow("camera", 780, 500);
		moveWindow("camera", 10000, 0);
		cv::imshow("camera", src);
		waitKey(1);
		
		if (isFirstFrame)
        {
            time_t now = time(0);
            tm *ltm = localtime(&now);
            char timestamp[20];
            sprintf(timestamp, "%04d-%02d-%02d_%02d:%02d:%02d", 1900 + ltm->tm_year, 1 + ltm->tm_mon, ltm->tm_mday,
                    ltm->tm_hour, ltm->tm_min, ltm->tm_sec);
            string videoFilename = "/home/stoair/Videos/" + string(timestamp) + ".avi";
            video.open(videoFilename, VideoWriter::fourcc('M', 'J', 'P', 'G'), 10, Size(src.cols, src.rows));
            if (!video.isOpened())
            {
                cerr << "Error: Failed to create video file: " << videoFilename << endl;
            }
            isFirstFrame = 0;
        }
        video.write(src);
	}
};

int main(int argc, char ** argv)
{
	rclcpp::init(argc, argv);
	auto node = std::make_shared<showImage>();
	rclcpp::spin(node);
	rclcpp::shutdown();
	return 0;
}
