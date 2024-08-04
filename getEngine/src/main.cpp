#include "model.h"
#include "cuda_utils.h"
#include "logging.h"
#include "types.h"

#include <chrono>
#include <cmath>
#include <iostream>
#include <ostream>
#include <fstream>

using namespace nvinfer1;

static Logger gLogger;
const static int kOutputSize = kMaxNumOutputBbox * sizeof(Detection) / sizeof(float) + 1;

bool serialize_engine(unsigned int max_batchsize, std::string& wts_name,
					  std::string& engine_name, float gd=0.33, float gw=0.50)
{
	// Create builder
    IBuilder* builder = createInferBuilder(gLogger);
    IBuilderConfig* config = builder->createBuilderConfig();

    // Create model to populate the network, then set the outputs and create an engine
    IHostMemory* serialized_engine = nullptr;
	serialized_engine = build_det_engine(max_batchsize, builder, config, DataType::kFLOAT, gd, gw, wts_name);

    // Serialize the engine
    assert(serialized_engine != nullptr);

    // Save engine to file
    std::ofstream p(engine_name, std::ios::binary);
    if (!p) {
        std::cerr << "Could not open plan output file" << std::endl;
        assert(false);
    }
    p.write(reinterpret_cast<const char*>(serialized_engine->data()), serialized_engine->size());

    // Close everything down
    delete serialized_engine;
    delete config;
    delete builder;
}


int main(int argc, char ** argv)
{
	cudaSetDevice(kGpuId);
	std::string wts_name = "../modules/circle.wts";
	std::string engine_name = "../engine/circle.engine";
	float gd = 0.0f, gw = 0.0f;
	gd = 0.33f;
    gw = 0.50f;
	if (serialize_engine(kBatchSize, wts_name, engine_name, gd, gw)) {
		std::cout << "Get engine successfully";
	} else {
		std::cout << "Fail to get engine";
	}
	return 0;
}
