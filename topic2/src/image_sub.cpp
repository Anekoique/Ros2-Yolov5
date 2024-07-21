#include "global.hpp"
#include "yolo.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "rclcpp/rclcpp.hpp"
#include "cv_bridge/cv_bridge.h"
#include "Mat2image.hpp"
#include "ros2_interfaces/msg/coord.hpp"
#include <ctime>

#define USE_CUDA true
using namespace Eigen;
using namespace std;
using namespace cv;

char coord[50];
char coord_last_frame[50];
int target_x = 317;
int target_y = 323;
int flag_servo = 0;
int count_servo = 0;
int count_nomove = 0;
int model_flag = 0;
string model_path = "/home/stoair/ros2CV/cv_ws/src/topic2/src/models/best_H.onnx";
string model_path_circle = "/home/stoair/ros2CV/cv_ws/src/topic2/src/models/best.onnx";
Matrix3d K; // 内参矩阵
Vector2d D; // 畸变矩阵

class imageSub : public rclcpp::Node
{
public:
    imageSub() : Node("webcam_sub") {
        std::cout << "OpenCV version: " << CV_VERSION << std::endl;
        readmodel();

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

private:
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscriber_;
    rclcpp::Publisher<ros2_interfaces::msg::Coord>::SharedPtr publisher_;
    rclcpp::TimerBase::SharedPtr timer_;
    VideoCapture cap;
    Yolo test;
    dnn::Net net1, net2;
    vector<Output> result;
    bool isFirstFrame = true;
    VideoWriter video;
    rclcpp::Time last_message_time_;
    bool message_received_;

    void readmodel() {
        if (test.readModel(net1, model_path, USE_CUDA))
            cout << "read net1 ok!" << endl;
        else
            cout << "read onnx1 model failed!";
        if (test.readModel(net2, model_path_circle, USE_CUDA))
            cout << "read net2 ok!" << endl;
        else
            cout << "read onnx2 model failed!";
    }

    void image_callback(const sensor_msgs::msg::Image::SharedPtr msg) {
        last_message_time_ = this->get_clock()->now();
        message_received_ = true;
        RCLCPP_INFO(this->get_logger(), "Receiving video frames");
        Mat src;
        std::string base64 = std::string((msg->data).begin(), (msg->data).end()); 
        Mat2Img::Base2Mat(base64, src);

        Mat img = handle_image(src, test);
        namedWindow("frame", WINDOW_NORMAL);
        cv::circle(img, cv::Point(target_x,target_y), 20, cv::Scalar(0,255,255), 2, cv::LINE_AA);// 黄色
        imshow("frame", img);
        resizeWindow("frame", 1040, 780);
        waitKey(1);

        drive_servo(coord, target_x, target_y, 3);
        if (strcmp(coord, coord_last_frame) == 0) { // 如Z果连续多帧坐标相同，说明没有检测到目标，不再发送坐标数据
            count_nomove++;
            if (count_nomove >= 30)
            {
                strcpy(coord, "0");
                count_nomove = 0;
            }
        }
        strcpy(coord_last_frame, coord);

        if (isFirstFrame)
        {
            time_t now = time(0);
            tm *ltm = localtime(&now);
            char timestamp[20];
            sprintf(timestamp, "%04d-%02d-%02d_%02d:%02d:%02d", 1900 + ltm->tm_year, 1 + ltm->tm_mon, ltm->tm_mday,
                    ltm->tm_hour, ltm->tm_min, ltm->tm_sec);
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

    void check_message_timeout() {
        if (!message_received_) {
            return;
        }

        rclcpp::Time current_time = this->get_clock()->now();
        if ((current_time - last_message_time_).seconds() >= 1.0) {
            // 如果3秒内没有接收到新的消息，执行操作
            RCLCPP_INFO(this->get_logger(), "No message received in the last 3 seconds. Executing operation.");

            // 执行操作...
            strcpy(coord, "0");
            // 重置标志
            message_received_ = false;
        }
    }

    Mat handle_image(Mat frame, Yolo& test)
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
        if (test.Detect(img, net1, result, 1)) {
            img = test.drawPred(img, result, color, 1);
        }
        if (test.Detect(img, net2, result, 0)) {
            test.target(img, result, 0);
            img = test.drawPred(img, result, color, 0);
        }
        return img;
    }

    void timer_callback() {
        //check_message_timeout();
        ros2_interfaces::msg::Coord message;
        sscanf(coord, "%d,%d,%d", &message.x, &message.y, &message.flag_servo);
        publisher_->publish(message);
        // RCLCPP_INFO(this->get_logger(), "publish coord");
    }

    void drive_servo(char *str, int target_x, int target_y, int times) {
        int x, y, r;
        int rmax = 20;
        if (sscanf(str, "%d,%d", &x, &y) == 2) {
            r = calculateDistance(x, y, target_x, target_y);

            if (r <= rmax)
                count_servo++;
            else {
                count_servo = 0;
                flag_servo = 0;
            }
            cout << r << "|" << count_servo << endl;
            if (count_servo >= times) {
                flag_servo = 1;
            }
        }
    }

    double calculateDistance(double x1, double y1, double x2, double y2) {
        double dx = x2 - x1;
        double dy = y2 - y1;
        double distance = std::sqrt(dx * dx + dy * dy);
        return distance;
    }
};



int main(int argc, char ** argv)
{
    coord[0] = '0';
    rclcpp::init(argc, argv);
    auto node = std::make_shared<imageSub>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;

}
