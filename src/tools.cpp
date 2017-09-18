#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> & estimations,
                              const vector<VectorXd> & ground_truth) 
{
    VectorXd rmse(4);
    rmse << 0,0,0,0;

    if (estimations.size() == 0 ||
    estimations.size() != ground_truth.size())
    {
        cout << "Invalid input vectors" << endl;
        return rmse;
    }

    //accumulate squared residuals
    VectorXd squaredSum(4);
    squaredSum << 0,0,0,0;
    
    for(int i = 0; i < estimations.size(); ++i)
    {
        
        VectorXd res = estimations[i] - ground_truth[i];
        VectorXd resSq = res.array() * res.array();
        squaredSum += resSq;
        
    }

    //calculate the mean
    squaredSum = squaredSum / estimations.size();

    //calculate the squared root
    rmse = squaredSum.array().sqrt();

    //return the result
    return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd & x_state) 
{
    MatrixXd Hj(3,4);
    //recover state parameters
    double px = x_state(0);
    double py = x_state(1);
    double vx = x_state(2);
    double vy = x_state(3);
    
    double g = px*px + py*py;

    //check division by zero
    if (g < 0.0001)
    {
        cout << "Cannot calculate Jacobian matrix. Invalid values." << endl;
        return Hj;
    }
    
    //compute the Jacobian matrix
    double g2 = sqrt(g);
    double g3 = pow(g2, 3.0);
    
    Hj << px/g2, py/g2, 0.0, 0.0,
          -py/g, px/g,  0.0, 0.0,
          py*(vx*py-vy*px)/g3, px*(vy*px-vx*py)/g3, px/g2, py/g2;

    return Hj;
}
