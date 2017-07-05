#include "DensityTree.h"
#include <iostream>
#include <random>
#include <functional>
#include <cmath>

using namespace cv;
using namespace std;
DensityTree::DensityTree(unsigned int D, unsigned int n_thresholds, Mat X) 
{
    this-> D=D;
    this-> X=X;
    this-> n_thresholds=n_thresholds;
}
void DensityTree::train()
{
    // train theta for each S, note for density tree all data is used for estimation
	// 
	int Rho = rand()*n_thresholds + 1;

	double theta = 0;
	double theta_tmp = 0;
	double max_info_gain = 0;
	while (theta_tmp < max(X, 0)) {
		Mat S_left = X(X(0, :) < theta_tmp);
		Mat S_right = X(X(1, :) >= theta_tmp);
		double info_gain = getInfoGain(S_left, S_right, S);
		if (info_gain > max_info_gain) {
			theta = theta_tmp;
			info_gain = max_info_gain;
		}
		theta_tmp += (double)max(X, 0) / Rho;
	}
	// load cluster result into container
	for (int i = 0;i < X.rows;i++) {
		if (X.at<double>(i, 0) < theta)  leftS.push_back(i);
		else rightS.push_back(i);
	}

    cout << "tree training completed" << endl;//Temporla
}
Mat DensityTree::densityXY()
{
    // using prediction model here
	// density estimation of gaussian distribution

    return X;//Temporal
}

double getInfoGain(Mat& SL, Mat& SR, Mat& S) {
	// calculation information gain
	// return (double) infoGain
	Mat CovarL, CovarR, CovarA ,mean;
	// calculation covariance matrix
	calcCovarMatrix(SL, CovarL, mean, CV_COVAR_NORMAL|CV_COVAR_ROWS);
	calcCovarMatrix(SR, CovarR, mean, CV_COVAR_NORMAL|CV_COVAR_ROWS);
	calcCovarMatrix(S, CovarA, mean, CV_COVAR_NORMAL|CV_COVAR_ROWS);
	// calculation determinant and information gain
	double info_gain = log(determinant(CovarA)) - (double)SL.rows / (double)S.rows*log(determinant(CovarL)) - (double)SR.rows / (double)S.rows*log(determinant(CovarR));
	
	return info_gain;
}
