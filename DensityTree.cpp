#include "DensityTree.h"
#include <iostream>
#include <random>
#include <functional>
#include <cmath>

using namespace cv;
using namespace cv::ml;
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

	vector<double> thres(n_thresholds);	//random thresholds in [xmin,xmax]
	double xmin, xmax;
	minMaxIdx(X, &xmin, &xmax);
	getRandomArray(thres, xmin, xmax);
	
	double theta_tmp = 0;
	double max_info_gain = 0;
	auto max_thres = thres.begin();
	auto iter = thres.begin();
	Mat SL, SR;
	while (iter != thres.end()) {
		// separate X into SL and SR
		SL.release();
		SR.release();
		double t = *iter;
		for (int i = 0;i < X.rows;i++) {
			if (X.at<double>(i, 0) < t) {
				SL.push_back(X.row(i));
			}
			else SR.push_back(X.row(i));
		}
		double info_gain = getInfoGain(SL, SR, X);
		if (info_gain > max_info_gain) {
			max_info_gain = info_gain;
			max_thres = iter;
		}

	}
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
	// train two GMM here

    cout << "tree training completed" << endl;//Temporla
}

Mat DensityTree::densityXY()
{
    // using prediction model here
	// density estimation of gaussian distribution


    return X;//Temporal
}

void DensityTree::getRandomArray(vector<double>& tar, const double & min, const double & max)
{
	// generate random double number between min and max.
	default_random_engine gen;
	uniform_real_distribution<double> distribution(min, max);
	auto dice = bind(distribution, gen);
	generate(tar.begin(), tar.end(), dice);
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

