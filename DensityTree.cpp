#include "DensityTree.h"
#include <iostream>
#include <random>
#include <functional>
#include <cmath>

using namespace cv;
//using namespace cv::ml;
using namespace std;

double getInfoGain(Mat&, Mat&, Mat&);

DensityTree::DensityTree(unsigned int D, unsigned int n_thresholds, Mat X) 
{
    this-> D=D;
    this-> X=X;
    this-> n_thresholds=n_thresholds;
}
void DensityTree::train()
{
    // train thetas for each node , note that for density tree all data is used for estimation
	// push each 
	int nodeCount = 0;
	bool isLeaf;
	vector<double> thres(n_thresholds);	//random thresholds in [xmin,xmax]
	double xmin, xmax;
	minMaxIdx(X, &xmin, &xmax);
	getRandomArray(thres, xmin, xmax);
	subsetBuffer.push_back(X);
	while(nodeCount!= pow(2,D)-1){
		// determine isLeaf
		if (nodeCount> pow(2,D-1)-2) isLeaf = true;
		else isLeaf = false;
		if(!isLeaf){	
			double theta_tmp = 0;
			double max_info_gain = 0;
			auto max_thres = thres.begin();
			auto iter = thres.begin();
			Mat SL, SR;
			Mat S = *subsetBuffer.begin();
			while (iter != thres.end()) {
				// separate X into SL and SR
				SL.release();
				SR.release();
				double t = *iter;
				for (int i = 0;i < X.rows;i++) {
					if (X.at<double>(i, 0) < t) {
						SL.push_back(S.row(i));
					}
					else SR.push_back(S.row(i));
				}
				double info_gain = getInfoGain(SL, SR, S);
				if (info_gain > max_info_gain) {
					max_info_gain = info_gain;
					max_thres = iter;
				}
			}
			// save the result of this training
			WeakLearner node;
			node.innerNode(*max_thres);
			nodeArray.push_back(node);
			// change subset buffer for next step of training
			subsetBuffer.push_back(SL);
			subsetBuffer.push_back(SR);
			subsetBuffer.erase(subsetBuffer.begin());
		}
		// calculate the gaussian distribution model in leaf nodes
		else{
			Mat S = *subsetBuffer.begin();
			Scalar meanX;
			Scalar meanY;
			Scalar sdX;
			Scalar sdY;
			meanStdDev(S.col(0),meanX,sdX);
			meanStdDev(S.col(0),meanY,sdY); 
			WeakLearner node;
			node.leafNode(meanX[0],sdX[0],meanY[0],sdY[0],S.rows);
			nodeArray.push_back(node);
			// update subset buffer
			subsetBuffer.erase(subsetBuffer.begin());
		}
		nodeCount++;
	}

    cout << "tree training completed" << endl;//Temporla
}

Mat DensityTree::densityXY()
{
	// density estimation of gaussian distribution
	Mat denXY = X;
	denXY.setTo(0);
	for(int i = 0; i < X.rows; i++){
		double tarX = denXY.at<double> (i,0);
		auto iter = nodeArray.begin();
		while(iter->isleafNode()==false){
			// test over density tree
			int thisIdx = distance(nodeArray.begin(),iter);
			if( tarX < iter->getTheta() ){
				// go left;
				iter = nodeArray.begin()+iter->getLeftIdx(thisIdx);
			}
			else{
				// go right;
				iter = nodeArray.begin()+iter->getRightIdx(thisIdx);
			}
			
		}
		// reach leaf node;
		denXY.at<double>(i,0) = iter->getDensity(denXY.at<double>(i,0),true);	// get X density
		denXY.at<double>(i,1) = iter->getDensity(denXY.at<double>(i,1),false); // get Y density
	}

    return denXY;//Temporal
}

void DensityTree::getRandomArray(vector<double>& tar, const double & min, const double & max)
{
	// generate random double number between min and max.
	default_random_engine gen;
	uniform_real_distribution<double> distribution(min, max);
	auto dice = bind(distribution, gen);
	generate(tar.begin(), tar.end(), dice);
}

WeakLearner::WeakLearner(){}

void WeakLearner::innerNode(double thetaIn){
	theta = thetaIn;
	isLeaf = false;
}

void WeakLearner::leafNode(double mx, double sx, double my, double sy,int n){
	meanX = mx;
	meanY = my;
	sdX = sx;
	sdY = sy;
	Num = n;
	isLeaf = true;
}

int WeakLearner::getLeftIdx(int thisIdx)
{
	return 2*thisIdx+1;
}

int WeakLearner::getRightIdx(int thisIdx)
{
	return 2*thisIdx+2;
}

bool WeakLearner::isleafNode(){return isLeaf;}
double WeakLearner::getTheta(){return theta;}
double WeakLearner::getDensity(double val, bool dim){
	if(dim == true){//X dim
		return (double)Num/1000.0*1.0/sqrt(2.0*PI*sdX)*exp(-0.5*pow(val-meanX,2)/sdX);
	}else{	// Y dim
		return (double)Num/1000.0*1.0/sqrt(2.0*PI*sdY)*exp(-0.5*pow(val-meanY,2)/sdY);
	}
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

