#include "global.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "rclcpp/rclcpp.hpp"
#include "cv_bridge/cv_bridge.h"

using namespace cv;

class imagePub : public rclcpp::Node
{
public:
    imagePub() : Node("image_pub")
    {
        cap.open(0, CAP_V4L2);
        cap.set(CAP_PROP_FOURCC, VideoWriter::fourcc('M','J','P','G'));
        cap.set(CAP_PROP_FRAME_WIDTH, 640);
        cap.set(CAP_PROP_FRAME_HEIGHT, 480);

        publisher_ = this->create_publisher<sensor_msgs::msg::Image>("image_raw", 10);

        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(10), 
            std::bind(&imagePub::timer_callback, this)
        );
    }
private:
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr publisher_;
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

            std::vector<unsigned char> buffer;
            imencode(".jpg", frame, buffer, {IMWRITE_JPEG_QUALITY, 95});
            int datalen = buffer.size();
            img_msg->step = datalen;
            img_msg->data = buffer;
            img_msg->encoding = "jpeg";
            
            publisher_->publish(*img_msg);
        } else {
            RCLCPP_ERROR(this->get_logger(), "Failed to capture frame");
        }
        RCLCPP_INFO(this->get_logger(), "Publishing video frame");
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
