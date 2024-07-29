#include "postprocess.h"
#include "logging.h"
#include "cuda_utils.h"
#include "preprocess.h"
#include "yololayer.h"

#include <chrono>
#include <cmath>
#include <iostream>
#include <fstream>
#include <cassert>
#include <cstring>

using namespace nvinfer1;

static Logger gLogger;
const static int kOutputSize = kMaxNumOutputBbox * sizeof(Detection) / sizeof(float) + 1;

void prepare_buffers(ICudaEngine* engine, float** gpu_input_buffer, float** gpu_output_buffer,
                     float** cpu_output_buffer) {
    assert(engine->getNbIOTensors() == 2);

    TensorIOMode input_mode = engine->getTensorIOMode(kInputTensorName);
    if (input_mode != TensorIOMode::kINPUT) {
        std::cerr << kInputTensorName << " should be input tensor" << std::endl;
        assert(false);
    }
    TensorIOMode output_mode = engine->getTensorIOMode(kOutputTensorName);
    if (output_mode != TensorIOMode::kOUTPUT) {
        std::cerr << kOutputTensorName << " should be output tensor" << std::endl;
        assert(false);
    }
    // Create GPU buffers on device
    CUDA_CHECK(cudaMalloc((void**)gpu_input_buffer, kBatchSize * 3 * kInputH * kInputW * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)gpu_output_buffer, kBatchSize * kOutputSize * sizeof(float)));

    *cpu_output_buffer = new float[kBatchSize * kOutputSize];
}

void deserialize_engine(std::string& engine_name, IRuntime** runtime, ICudaEngine** engine,
                        IExecutionContext** context) {
    std::ifstream file(engine_name, std::ios::binary);
    if (!file.good()) {
        std::cerr << "read " << engine_name << " error!" << std::endl;
        assert(false);
    }
    size_t size = 0;
    file.seekg(0, file.end);
    size = file.tellg();
    std::cout << "size: " << size << std::endl;
    file.seekg(0, file.beg);
    char* serialized_engine = new char[size];
    assert(serialized_engine);
    file.read(serialized_engine, size);
    file.close();
    *runtime = createInferRuntime(gLogger);
    assert(*runtime);
    *engine = (*runtime)->deserializeCudaEngine(serialized_engine, size);
    cv::Mat img = cv::imread("../images/circle.jpg");
    std::cout << "read successfully" << std::endl;
    assert(*engine);
    *context = (*engine)->createExecutionContext();
    assert(*context);
    delete[] serialized_engine;
}

void infer(IExecutionContext& context, cudaStream_t& stream, void** gpu_buffers, float* output, int batchsize) {
    context.setInputTensorAddress(kInputTensorName, gpu_buffers[0]);
    context.setOutputTensorAddress(kOutputTensorName, gpu_buffers[1]);
    context.enqueueV3(stream);
    CUDA_CHECK(cudaMemcpyAsync(output, gpu_buffers[1], batchsize * kOutputSize * sizeof(float), cudaMemcpyDeviceToHost,
                               stream));
    cudaStreamSynchronize(stream);
}

int main(int argc, char ** argv)
{
    // Set GPU device and read engine file
    cudaSetDevice(kGpuId);
    std::string engine_name = "../engine/myEngine.engine";

    // Deserialize the engine from file
    IRuntime* runtime = nullptr;
    ICudaEngine* engine = nullptr;
    IExecutionContext* context = nullptr;
    deserialize_engine(engine_name, &runtime, &engine, &context);

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Init CUDA preprocessing
    cuda_preprocess_init(kMaxInputImageSize);

    // Prepare cpu and gpu buffers
    float* gpu_buffers[2];
    float* cpu_output_buffer = nullptr;
    prepare_buffers(engine, &gpu_buffers[0], &gpu_buffers[1], &cpu_output_buffer);

    cv::Mat img = cv::imread("../images/test.jpg");
    std::cout << "read successfully" << std::endl;
    if (img.empty()) {
        std::cerr << "read image error!" << std::endl;
        return -1;
    }

    // Preprocess
    cuda_preprocess(img.ptr(), img.cols, img.rows, gpu_buffers[0], kInputW, kInputH, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Run inference
    auto start = std::chrono::system_clock::now();
    infer(*context, stream, (void**)gpu_buffers, cpu_output_buffer, kBatchSize);
    auto end = std::chrono::system_clock::now();
    std::cout << "inference time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
                << "ms" << std::endl;

    // NMS
    std::vector<Detection> res_batch;
    nms(res_batch, cpu_output_buffer, kConfThresh, kNmsThresh);

    // print results
    for (size_t k = 0; k < res_batch.size(); k++) {
        std::cout  << " bbox: " << res_batch[k].bbox[0] << ", "
                   << res_batch[k].bbox[1] << ", " << res_batch[k].bbox[2] << ", "
                   << res_batch[k].bbox[3] << ", conf: " << res_batch[k].conf
                   << ", class_id: " << res_batch[k].class_id << std::endl;
    }

    // Draw bounding boxes
    draw_bbox(img, res_batch);

    cv::imshow("result", img);
    cv::waitKey(0);

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CUDA_CHECK(cudaFree(gpu_buffers[0]));
    CUDA_CHECK(cudaFree(gpu_buffers[1]));
    delete[] cpu_output_buffer;
    cuda_preprocess_destroy();
    // Destroy the engine
    delete context;
    delete engine;
    delete runtime;
}