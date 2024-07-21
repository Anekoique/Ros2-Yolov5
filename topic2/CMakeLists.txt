cmake_minimum_required(VERSION 3.8)
project(topic2)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()
list(APPEND CMAKE_PREFIX_PATH /)

# find dependencies

find_package(OpenCV 4.9.0 REQUIRED)
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(ros2_interfaces REQUIRED)
find_package(sensor_msgs REQUIRED)

include_directories( "/usr/include/eigen3" )
# include_directories( "/home/stoair/ros2CV/cv_ws/src/topic2/include/topic2")
LINK_DIRECTORIES("/opt/ros2/humble/lib")

# Declare executables and set dependencies
add_executable(image_pub src/image_pub.cpp src/image_base64.cpp)
ament_target_dependencies(image_pub rclcpp OpenCV cv_bridge ros2_interfaces sensor_msgs)
add_executable(image_sub src/image_sub.cpp src/yolo.cpp src/image_base64.cpp)
ament_target_dependencies(image_sub rclcpp OpenCV cv_bridge ros2_interfaces sensor_msgs)
install(TARGETS
  image_pub
  image_sub
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
