#pragma once

#define USE_FP32  // set USE_INT8 or USE_FP16 or USE_FP32

const static char* kInputTensorName = "data";
const static char* kOutputTensorName = "prob";
const static char* kProtoTensorName = "proto";

constexpr static int kNumClass = 1;

constexpr static int kBatchSize = 1;

constexpr static int kInputH = 480;
constexpr static int kInputW = 640;

constexpr static int kMaxNumOutputBbox = 1000;
constexpr static int kNumAnchor = 3;
constexpr static float kIgnoreThresh = 0.1f;

const static float kNmsThresh = 0.45f;
const static float kConfThresh = 0.5f;

const static int kGpuId = 0;

const static int kMaxInputImageSize = 4096 * 3112;
