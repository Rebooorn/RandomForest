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
#include "opencv2/ml/ml.hpp"
#include <vector>
#include <random>
#include <functional>

using namespace cv;
using namespace std;

#ifndef DENSITYTREE_H
#define DENSITYTREE_H
# define PI 3.14159265358979323846

class WeakLearner;
class DensityTree 
{
public:
    DensityTree();
    DensityTree(unsigned int D, unsigned int R, Mat X);
    void train();
    Mat densityXY();
	friend double getInfoGain(Mat& SL, Mat& SR, Mat& S);
	void getRandomArray(vector<double>& tar ,const double& min,const double& max);
private:
    unsigned int D;	//tree depth
    unsigned int n_thresholds;
    Mat X;
	vector<WeakLearner> nodeArray;
	vector<Mat> subsetBuffer;
   // auto dice;	// random number generator
	vector<int> leftS;
	vector<int> rightS;
};

class WeakLearner
{
public:
	WeakLearner();
	//~WeakLearner();
	void innerNode(double);
	void leafNode(double,double,double,double,int);
	int getLeftIdx(int);	//get index of left and right child;
	int getRightIdx(int);
	bool isleafNode();
	double getTheta();
	double getDensity(double,bool);		// true for X, false for Y
private:
	bool isLeaf = false;	
	double theta = 0;
	double meanX = 0;
	double meanY = 0;
	double sdX = 0;
	double sdY = 0;
	double Num = 0;	//used for only leaf nodes
	
};


#endif /* DENSITYTREE_H */

