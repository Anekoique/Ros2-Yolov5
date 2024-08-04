cmake_minimum_required(VERSION 3.10)
project(trtdet)

add_definitions(-std=c++11)
add_definitions(-DAPI_EXPORTS)
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CUDA_ARCHITECTURES 89)

#if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
#  add_compile_options(-Wall -Wextra -Wpedantic)
#endif()
list(APPEND CMAKE_PREFIX_PATH /)

set(CMAKE_CUDA_COMPILER /usr/local/cuda-12.1/bin/nvcc)
enable_language(CUDA)

find_package(OpenCV 4.9.0 REQUIRED)
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(ros2_interfaces REQUIRED)
find_package(sensor_msgs REQUIRED)

include_directories( "/usr/include/eigen3" )
link_directories("/opt/ros2/humble/lib")

include_directories(/usr/local/cuda-12.1/include)
link_directories(/usr/local/cuda-12.1/lib64)

include_directories(/home/stoair/packages/TensorRT-10.2.0.19/include)
link_directories(/home/stoair/packages/TensorRT-10.2.0.19/lib/)

include_directories(${PROJECT_SOURCE_DIR}/include/)
include_directories(${PROJECT_SOURCE_DIR}/plugin/)
file(GLOB_RECURSE SRCS ${PROJECT_SOURCE_DIR}/src/*.cpp ${PROJECT_SOURCE_DIR}/src/*.cu)
file(GLOB_RECURSE PLUGIN_SRCS ${PROJECT_SOURCE_DIR}/plugin/*.cu)

add_definitions(-DAPI_EXPORTS)
add_library(myplugins SHARED ${PLUGIN_SRCS})
target_link_libraries(myplugins nvinfer cudart)

add_executable(image_sub ./src/calibrator.cpp ./src/image_sub.cpp ./src/model.cpp ./src/postprocess.cpp 
				./src/preprocess.cu ./src/trtInfer.cpp)
ament_target_dependencies(image_sub rclcpp OpenCV cv_bridge ros2_interfaces sensor_msgs)
target_link_libraries(image_sub nvinfer cudart myplugins)
add_executable(image_pub src/image_pub.cpp)
ament_target_dependencies(image_pub rclcpp OpenCV cv_bridge sensor_msgs)
install(TARGETS
  image_sub
  image_pub
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
