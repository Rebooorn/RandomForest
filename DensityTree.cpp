#include "DensityTree.h"
#include <iostream>
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
    cout << "Not implemented" << endl;//Temporla
}
Mat DensityTree::densityXY()
{
    
    return X;//Temporal
}


