#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>
#include <cassert>
#include <cmath>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/*
 * Constructor.
 */
FusionEKF::FusionEKF() 
{
    is_initialized_ = false;

    previous_timestamp_ = 0;

    // initializing matrices
    R_laser_ = MatrixXd(2, 2);
    R_radar_ = MatrixXd(3, 3);
    H_laser_ = MatrixXd(2, 4);

    //measurement covariance matrix - laser
    R_laser_ << 0.0225, 0,
        0, 0.0225;

    //measurement covariance matrix - radar
    R_radar_ << 0.09, 0, 0,
        0, 0.0009, 0,
        0, 0, 0.09;

    H_laser_ << 1, 0, 0, 0,
        0, 1, 0, 0;

    noise_ax = 9.0f; 
    noise_ay = 9.0f;

}

/**
* Destructor.
*/
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage & measurement_pack) 
{

  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
    if (!is_initialized_) 
    {
        // first measurement
        cout << "EKF: Initializing" << endl;
        ekf_.x_ = VectorXd(4);
        ekf_.x_ << 1, 1, 1, 1;

        if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) 
        {
            ekf_.x_ = polarToCartesian(measurement_pack.raw_measurements_);
        }
        else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) 
        {
            ekf_.x_ << measurement_pack.raw_measurements_[0], 
                measurement_pack.raw_measurements_[1], 
                0, 
                0;
        }

        // done initializing, no need to predict or update
        previous_timestamp_ = measurement_pack.timestamp_;
        is_initialized_ = true;
        return;
    }

    /*****************************************************************************
    *  Prediction
    ****************************************************************************/

     //compute the time elapsed between the current and previous measurements
    float dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0; //dt - expressed in seconds
    previous_timestamp_ = measurement_pack.timestamp_;
    

    //Modify the F matrix so that the time is integrated
    ekf_.F_(0, 2) = dt;
    ekf_.F_(1, 3) = dt;
    
    //Update the process covariance matrix Q
    MatrixXd G(4, 2);
    G << dt*dt/2.0, 0, 
        0, dt*dt/2.0, 
        dt, 0, 
        0, dt;
        
    MatrixXd Qv(2, 2);
    Qv << noise_ax, 0, 
          0, noise_ay;
    
    ekf_.Q_ = G * Qv * G.transpose();

    ekf_.Predict();

    /*****************************************************************************
    *  Update
    ****************************************************************************/

    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) 
    {
        // Radar updates
        ekf_.R_ = R_radar_;
        ekf_.UpdateEKF(measurement_pack.raw_measurements_);

    } 
    else 
    {
        // Laser updates
        ekf_.R_ = R_laser_;
        ekf_.H_ = H_laser_;
        ekf_.Update(measurement_pack.raw_measurements_);
    }

    // print the output
    cout << "x_ = " << ekf_.x_ << endl;
    cout << "P_ = " << ekf_.P_ << "\n" << endl;
}

VectorXd 
FusionEKF::polarToCartesian(const VectorXd & inPolar)
{
    assert(inPolar.size() >= 2);
    
    VectorXd output(4);

    double rho = inPolar(0);
    double phi = inPolar(1);

    double px = rho * std::cos(phi);
    double py = -rho * std::sin(phi);

    output << px, py, 0.0, 0.0;

    return output;

}
