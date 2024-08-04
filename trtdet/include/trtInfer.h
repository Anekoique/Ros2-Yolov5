#ifndef __TRTINFER_H__
#define __TRTINFER_H__

#include "cuda_utils.h"
#include "logging.h"
#include "model.h"
#include "postprocess.h"
#include "preprocess.h"
#include "utils.h"
#include "global.hpp"

#include <chrono>
#include <cmath>
#include <iostream>
using namespace nvinfer1;
static Logger gLogger;
class TrtInfer
{
public:
    TrtInfer(){};
    void init(std::string& engine_file, int flag);
    ~TrtInfer();
    cv::Mat trtInfer(cv::Mat& img);
private:
    int type;
    std::string engine_file;
    nvinfer1::IRuntime* runtime;
    nvinfer1::ICudaEngine* engine;
    nvinfer1::IExecutionContext* context;
    cudaStream_t stream;
    float* gpu_buffers[2];
    float* cpu_output_buffer;
    const static int kOutputSize = kMaxNumOutputBbox * sizeof(Detection) / sizeof(float) + 1;;
    void prepare_buffers(ICudaEngine* engine, float** gpu_input_buffer, 
                         float** gpu_output_buffer,float** cpu_output_buffer);
    void infer(IExecutionContext& context, cudaStream_t& stream, void** gpu_buffers,
               float* output, int batchsize);
    void deserialize_engine(std::string& engine_name, IRuntime** runtime, ICudaEngine** engine,
                            IExecutionContext** context);
    void find_target(std::vector<Detection>& res);

};

#endif // __TRTINFER_H__
