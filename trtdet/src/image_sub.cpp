#include "global.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "rclcpp/rclcpp.hpp"
#include "cv_bridge/cv_bridge.h"
#include "ros2_interfaces/msg/coord.hpp"
#include "trtInfer.h"

using namespace Eigen;
using namespace nvinfer1;
using namespace cv;
using namespace std;

char cir_coord[50];
char H_coord[50];
char cir_last_frame[50];
char H_last_frame[50];
double target_x = 317;
double target_y = 338;
int flag_servo = 0;
int cir_count = 0;
int H_count = 0;
int model_flag = 0;
Matrix3d K; // 内参矩阵
Vector2d D; // 畸变矩阵
std::string circle_engine_path = "/home/stoair/ros2CV/cv_ws/src/trtdet/engine/circle.engine";
std::string H_engine_path = "/home/stoair/ros2CV/cv_ws/src/trtdet/engine/H.engine";

class imageSub : public rclcpp::Node
{
public:
    imageSub() : Node("webcam_sub") {
        isFirstFrame = true;
        cir_det.init(circle_engine_path, 0);
        H_det.init(H_engine_path, 1);

        start_time_ = this->get_clock()->now();

        subscriber_ = this->create_subscription<sensor_msgs::msg::Image>(
            "image_raw", 
            10, 
            std::bind(&imageSub::image_callback, this, std::placeholders::_1)
        );

        publisher_ = this->create_publisher<ros2_interfaces::msg::Coord>("coord", 10);
        
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(10), 
            std::bind(&imageSub::timer_callback, this)
        );

    }
    ~imageSub() {
        if (video.isOpened())
        {
            video.release();
        }
        
    }

private:
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscriber_;
    rclcpp::Publisher<ros2_interfaces::msg::Coord>::SharedPtr publisher_;
    rclcpp::TimerBase::SharedPtr timer_;
    cv::VideoCapture cap;
    cv::VideoWriter video;
    rclcpp::Time last_message_time_, start_time_;
    bool isFirstFrame;
    TrtInfer cir_det;
    TrtInfer H_det;

    void image_callback(const sensor_msgs::msg::Image::SharedPtr msg) {
        //RCLCPP_INFO(this->get_logger(), "Receiving video frames");

        vector<unsigned char> buff = msg->data;
        const int datalen = msg->step;
        unsigned char *new_buff;
        new_buff = new unsigned char[datalen];
        for (int i = 0; i < datalen; i++) {
            new_buff[i] = buff[i];
        }
        Mat src = imdecode(Mat(1, datalen, CV_8UC1, new_buff), IMREAD_COLOR);
        Mat img = handle_image(src);

        namedWindow("frame", WINDOW_NORMAL);
        cv::circle(img, cv::Point(target_x,target_y), 20, cv::Scalar(0,255,255), 2, cv::LINE_AA);
		resizeWindow("frame", 640, 640);
		moveWindow("frame", 10000, 10000);
        cv::imshow("frame", img);
        waitKey(1);

        drive_servo(cir_coord, target_x, target_y, 1);
        if (strcmp(cir_coord, cir_last_frame) == 0) { 
            cir_count++;
            if (cir_count >= 30)
            {
                strcpy(cir_coord, "0");
                cir_count = 0;
            }
        }
        if (strcmp(H_coord, H_last_frame) == 0) { 
            H_count++;
            if (H_count >= 150)
            {
                strcpy(H_coord, "0");
                H_count = 0;
            }
        }
        
        strcpy(cir_last_frame, cir_coord);
        strcpy(H_last_frame, H_coord);

        if (isFirstFrame)
        {
            time_t now = time(0);
            tm *ltm = localtime(&now);
            char timestamp[20];
            sprintf(timestamp, "%04d-%02d-%02d_%02d:%02d:%02d", 1900 + ltm->tm_year, 1 + ltm->tm_mon, ltm->tm_mday, ltm->tm_hour, ltm->tm_min, ltm->tm_sec);
            string videoFilename = "/home/stoair/Videos/" + string(timestamp) + ".avi";
            video.open(videoFilename, VideoWriter::fourcc('M', 'J', 'P', 'G'), 10, Size(img.cols, img.rows));
            if (!video.isOpened())
            {
                cerr << "Error: Failed to create video file: " << videoFilename << endl;
            }
            isFirstFrame = false;
        }
        video.write(img);
    }


    Mat handle_image(Mat frame)
    {
        cv::Mat K = (cv::Mat_<double>(3, 3) << 514.0045, 0, 321.6074,
                                               0, 514.6655, 260.0872,
                                               0, 0, 1);
        cv::Mat D = (cv::Mat_<double>(4, 1) << 0.1631, -0.2023, 0, 0);
        vector<Scalar> color;

        srand(time(0));
        for (int i = 0; i < 80; i++) {
            int b = rand() % 256;
            int g = rand() % 256;
            int r = rand() % 256;
            color.push_back(Scalar(b, g, r));
        }

        Mat img;
        undistort(frame, img, K, D, K);
        Mat cir_img = cir_det.trtInfer(img);
        Mat H_img = H_det.trtInfer(img);
        return H_img;
    }

    void timer_callback() {
        ros2_interfaces::msg::Coord message;
        sscanf(cir_coord, "%f,%f,%d", &message.x1, &message.y1, &message.flag_servo);
        sscanf(H_coord, "%f, %f", &message.x2, &message.y2);
        RCLCPP_INFO(this->get_logger(), "Publishing Coord, %f, %f, %f, %f, %d", message.x1, message.y1, message.x2, message.y2, message.flag_servo);
        publisher_->publish(message);
    }

    void drive_servo(char *str, double target_x, double target_y, double duration) {
        float x, y;
        double r;
        float rmax = 20;
        static bool start_timing = false;
        if (sscanf(str, "%f,%f", &x, &y) == 2) {
            r = calculateDistance(x, y, target_x, target_y);
            if (r <= rmax)
                if (start_timing == false) {
                    start_timing = true;
                    last_message_time_ = this->get_clock()->now();
                    RCLCPP_INFO(this->get_logger(), "begin to timing");
                } else {
                    rclcpp::Time current_time = this->get_clock()->now();
                    if ((current_time - last_message_time_).seconds() >= duration) {
                        RCLCPP_INFO(this->get_logger(), "flag_servo has set to 1");
                        flag_servo = 1;
                    }
                }
            else {
                start_timing = false;
                flag_servo = 0;
            }
        }
    }

    double calculateDistance(float x1, float y1, double x2, double y2) {
        double dx = x2 - x1;
        double dy = y2 - y1;
        double distance = std::sqrt(dx * dx + dy * dy);
        return distance;
    }
};

int main(int argc, char ** argv)
{
    cir_coord[0] = '0';
    H_coord[0] = '0';
    rclcpp::init(argc, argv);
    auto node = std::make_shared<imageSub>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
