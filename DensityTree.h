/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   DensityTree.h
 * Author: dalia
 *
 * Created on June 19, 2017, 12:30 AM
 */

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <vector>

using namespace cv;
using namespace std;

#ifndef DENSITYTREE_H
#define DENSITYTREE_H

class DensityTree 
{
public:
    DensityTree();
    DensityTree(unsigned int D, unsigned int R, Mat X);
    void train();
    Mat densityXY();
private:
    unsigned int D;
    unsigned int n_thresholds;
    Mat X;
  //  auto dice;	// random number generator
};

#endif /* DENSITYTREE_H */

