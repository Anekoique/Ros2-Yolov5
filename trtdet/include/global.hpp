#ifndef GLOBAL_H
#define GLOBAL_H

#include <Eigen/Core>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

using namespace Eigen;

extern char cir_coord[50];
extern char H_coord[50];
extern int flag_servo;
extern double target_x;
extern double target_y;
extern int model_flag;
extern Matrix3d K;
extern Vector2d D;

#endif
