#include "DensityTree.h"
#include <iostream>
#include <random>

using namespace cv;
using namespace std;
DensityTree::DensityTree(unsigned int D, unsigned int n_thresholds, Mat X) 
{
    this-> D=D;
    this-> X=X;
    this-> n_thresholds=n_thresholds;
    
    std::default_random_engine generator;
    std::uniform_int_distribution<int> distribution(1,n_thresholds);
    dice = std::bind(distribution
    
}
void DensityTree::train()
{
    // train theta for each S, note for density tree all data is used for estimation
    
    cout << "Not implemented" << endl;//Temporla
}
Mat DensityTree::densityXY()
{
    
    return X;//Temporal
}



