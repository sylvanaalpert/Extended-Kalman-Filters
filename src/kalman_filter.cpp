#include "kalman_filter.h"
#include "tools.h"
#include <cmath>
#include <cassert>

using Eigen::MatrixXd;
using Eigen::VectorXd;

KalmanFilter::KalmanFilter() 
{
    //state covariance matrix P
    P_ = MatrixXd(4, 4);
    P_ << 1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1000, 0,
        0, 0, 0, 1000;

    //the initial transition matrix F_
    F_ = MatrixXd(4, 4);
    F_ << 1, 0, 1, 0,
        0, 1, 0, 1,
        0, 0, 1, 0,
        0, 0, 0, 1;
}

KalmanFilter::~KalmanFilter() {}

void 
KalmanFilter::Predict() 
{
    x_ = F_ * x_;
    MatrixXd Ft = F_.transpose();
    P_ = F_ * P_ * Ft + Q_;
}

void 
KalmanFilter::update_priv(const VectorXd & inY)
{
    MatrixXd Ht = H_.transpose();
    MatrixXd S = H_ * P_ * Ht + R_;
    MatrixXd Si = S.inverse();
    MatrixXd PHt = P_ * Ht;
    MatrixXd K = PHt * Si;

    //new estimate
    x_ = x_ + (K * inY);
    long x_size = x_.size();
    MatrixXd I = MatrixXd::Identity(x_size, x_size);
    P_ = (I - K * H_) * P_;
}

void 
KalmanFilter::Update(const VectorXd & z) 
{
    VectorXd z_pred = H_ * x_;
    VectorXd y = z - z_pred;
    update_priv(y);
    
}

void 
KalmanFilter::UpdateEKF(const VectorXd & z) 
{
    H_ = Tools::CalculateJacobian(x_);
    VectorXd z_pred = cartesianToPolar(x_);
    VectorXd y = z - z_pred;

    y[1]  = normalizeAngle(y[1]);
    update_priv(y);

}

float 
KalmanFilter::normalizeAngle(float inAngle)
{
    float outAngle = inAngle;

    float min = -M_PI;
    float max = M_PI;

    if  (inAngle < min)
    {
        outAngle = max + std::fmod(inAngle - min, max - min);
    }
    else
    {
        outAngle = min + std::fmod(inAngle - min, max - min);
    }

    return outAngle;
}

VectorXd 
KalmanFilter::cartesianToPolar(VectorXd & inCartesian)
{
    assert(inCartesian.size() == 4);
    
    VectorXd outPolar(3);
    outPolar << 0, 0, 0;

    double px = inCartesian(0);
    double py = inCartesian(1);
    double vx = inCartesian(2);
    double vy = inCartesian(3);

    double a = std::sqrt(std::pow(px, 2) + std::pow(py, 2));
    if (a < 0.00001)
    {
        return outPolar;
    }

    double b = std::atan2(py, px);

    double c = (px*vx + py*vy) / a;

    outPolar << a, b, c;

    return outPolar;
}
