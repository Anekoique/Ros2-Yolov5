#ifndef __TRTINFER_H__
#define __TRTINFER_H__

#include "cuda_utils.h"
#include "logging.h"
#include "model.h"
#include "postprocess.h"
#include "preprocess.h"
#include "utils.h"

#include <chrono>
#include <cmath>
#include <iostream>

using namespace nvinfer1;

class TrtInfer
{
public:
    TrtInfer::TrtInfer(){};
    void init(std::string& engine_file);
    ~TrtInfer();
    cv::Mat trtInfer(cv::Mat& img);
private:
    std::string engine_file;
    Iruntime* runtime;
    ICudaEngine* engine;
    IExecutionContext* context;
    cudaStream_t stream;
    float* gpu_buffers[2];
    float* cpu_output_buffer;
    static Logger gLogger;
    const static int kOutputSize;
    void prepare_buffers(ICudaEngine* engine, float** gpu_input_buffer, 
                         float** gpu_output_buffer,float** cpu_output_buffer);
    void infer(IExecutionContext& context, cudaStream_t& stream, void** gpu_buffers,
               float* output, int batchsize);
    void deserialize_engine(std::string& engine_name, IRuntime** runtime, ICudaEngine** engine,
                            IExecutionContext** context);

}

#endif // __TRTINFER_H__