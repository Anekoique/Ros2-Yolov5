cmake_minimum_required(VERSION 3.10)

project(GET_ENGINE)

add_definitions(-std=c++11)
add_definitions(-DAPI_EXPORTS)
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CUDA_ARCHITECTURES 89)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

# TODO(Call for PR): make cmake compatible with Windows
set(CMAKE_CUDA_COMPILER /usr/local/cuda-12.1/bin/nvcc)
enable_language(CUDA)

# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
# cuda
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

find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(get_engine ${SRCS})
target_link_libraries(get_engine nvinfer)
target_link_libraries(get_engine cudart)
target_link_libraries(get_engine myplugins)
target_link_libraries(get_engine ${OpenCV_LIBS})
