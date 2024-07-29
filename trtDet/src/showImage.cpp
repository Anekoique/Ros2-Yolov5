#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "cv_bridge/cv_bridge.h"
#include "opencv2/opencv.hpp"
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

	void image_callback(const sensor_msgs::msg::Image::SharedPtr msg) {
		RCLCPP_INFO(this->get_logger(), "Receiving video frames");
		vector<unsigned char> buff = msg->data;
        const int datalen = msg->step;
        unsigned char *new_buff;
        new_buff = new unsigned char[datalen];
        for (int i = 0; i < datalen; i++) {
            new_buff[i] = buff[i];
        }
        Mat src = imdecode(Mat(1, datalen, CV_8UC1, new_buff), IMREAD_COLOR);
		namedWindow("camera", WINDOW_NORMAL);
		resizeWindow("camera", 780, 500);
		moveWindow("camera", 10000, 0);
		cv::imshow("camera", src);
		waitKey(1);
		
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
