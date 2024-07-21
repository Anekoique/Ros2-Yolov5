#include "global.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "rclcpp/rclcpp.hpp"
#include "cv_bridge/cv_bridge.h"
#include "ros2_interfaces/msg/coord.hpp"
#include "Mat2image.hpp"

using namespace cv;

char coord[50];

class imagePub : public rclcpp::Node
{
public:
    imagePub() : Node("image_pub")
    {
        publisher_ = this->create_publisher<sensor_msgs::msg::Image>("image_raw", 10);

        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(10), 
            std::bind(&imagePub::timer_callback, this)
        );

        subscriber_ = this->create_subscription<ros2_interfaces::msg::Coord>(
            "coord", 
            10, 
            std::bind(&imagePub::coord_callback, this, std::placeholders::_1)
        );
        cap.open(0, CAP_V4L2);
        cap.set(CAP_PROP_FOURCC, VideoWriter::fourcc('M','J','P','G'));
        cap.set(CAP_PROP_FRAME_WIDTH, 640);
        cap.set(CAP_PROP_FRAME_HEIGHT, 480);
    }
private:
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr publisher_;
    rclcpp::Subscription<ros2_interfaces::msg::Coord>::SharedPtr subscriber_;
    rclcpp::TimerBase::SharedPtr timer_;
    VideoCapture cap;
    int jpeg_quality_ = 95;

    void timer_callback()
    {
        Mat frame;
        cap >> frame;
        if (frame.rows > 0 && frame.cols > 0)
        {
            auto img_msg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", frame).toImageMsg();
            std::string base64;
            Mat2Img::Mat2Base64(frame, base64);
            std::vector<unsigned char> ucharvector = std::vector<unsigned char>(base64.begin(), base64.end());
            img_msg->data = ucharvector;
            publisher_->publish(*img_msg);
            RCLCPP_INFO(this->get_logger(), "Publishing video frame");
        }
        else {
            RCLCPP_INFO(this->get_logger(), "Camera has problem, publish failed");
        }
    }

    void coord_callback(const ros2_interfaces::msg::Coord::SharedPtr msg)
    {
        int x = msg->x;
        int y = msg->y;
        int flag = msg->flag_servo;
        RCLCPP_INFO(this->get_logger(), "收到坐标(%d, %d), flag_servo = %d", x, y, flag);
    }
    
};

int main(int argc, char ** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<imagePub>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
