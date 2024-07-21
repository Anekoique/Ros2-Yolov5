#ifndef GLOBAL_H
#define GLOBAL_H

#include <Eigen/Core>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

using namespace Eigen;

extern char coord[50];
extern int flag_servo;
extern int target_x;
extern int target_y;
extern int model_flag;
extern Matrix3d K;
extern Vector2d D;
char *getUndistortedPixelCoord(double x, double y);

#endif