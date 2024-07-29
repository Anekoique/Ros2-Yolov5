#include "trtInfer.h"

using namespace nvinfer1;

TrtInfer::kOutputSize = kMaxNumOutputBbox * sizeof(Detection) / sizeof(float) + 1;

void TrtInfer::init(const std::string& engine_file)
{
    this->engine_file = engine_file;
    this->runtime = nullptr;
    this->engine = nullptr;
    this->context = nullptr;
    this->cpu_output_buffer = nullptr;

    deserialize_engine(engine_file, &runtime, &engine, &context);

    CUDA_CHECK(cudaStreamCreate(&stream));
    cuda_preprocess_init(kMaxInputImageSize);
    prepare_buffers(engine, &gpu_buffers[0],&gpu_buffers[1], &cpu_output_buffer);
}

TrtInfer::~TrtInfer()
{
    // Release stream and buffers
    cudaStreamDestroy(stream);
    CUDA_CHECK(cudaFree(gpu_buffers[0]));
    CUDA_CHECK(cudaFree(gpu_buffers[1]));
    delete[] this->cpu_output_buffer;
    cuda_preprocess_destroy();
    // Destroy the engine
    delete this->context;
    delete this->engine;
    delete this->runtime;
}

void TrtInfer::prepare_buffers(ICudaEngine* engine, float** gpu_input_buffer, float** gpu_output_buffer,
                     float** cpu_output_buffer) {
    assert(engine->getNbIOTensors() == 2);
    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
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

void TrtInfer::deserialize_engine(std::string& engine_name, IRuntime** runtime, ICudaEngine** engine,
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
    assert(*engine);
    *context = (*engine)->createExecutionContext();
    assert(*context);
    delete[] serialized_engine;
}

void TrtInfer::infer(IExecutionContext& context, cudaStream_t& stream, void** gpu_buffers, float* output, int batchsize) {
    context.setInputTensorAddress(kInputTensorName, gpu_buffers[0]);
    context.setOutputTensorAddress(kOutputTensorName, gpu_buffers[1]);
    context.enqueueV3(stream);
    CUDA_CHECK(cudaMemcpyAsync(output, gpu_buffers[1], batchsize * kOutputSize * sizeof(float), cudaMemcpyDeviceToHost,
                               stream));
    cudaStreamSynchronize(stream);
}

Mat TrtInfer::trtInfer(std::Mat& img) {
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
    std::vector<Detection> res;
    nms(res, cpu_output_buffer, kConfThresh, kNmsThresh);

    // Draw bounding boxes
    draw_bbox(img, res);

    return img;
}