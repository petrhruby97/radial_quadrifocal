#ifndef RQT_SOLVER_H_
#define RQT_SOLVER_H_

#include <Eigen/Core>
#include <Eigen/Dense>
#include "types.h"

namespace rqt {

// finds the radial cameras that are consistent with the input files
// returns the number of found relative poses
// the output vectors P1_out, P2_out, P3_out, P4_out, cam2QF, and Xs should be initialized to a maximal possible number
// of feasible output poses for every i from 0 to return_value-1: cameras P1_out[i], P2_out[i], P3_out[i], P4_out[i] + 3D
// points Xs[i] give a configuration that is consistent with the input
int radial_quadrifocal_solver(
    const std::vector<std::complex<double>>
        start_problem, // coefficients of the starting problem (loaded with load_start_system)
    const std::vector<std::vector<std::complex<double>>>
        start_sols,                          // solutions to the starting problem (loaded with load_start_system)
    const std::vector<Eigen::Vector2d> &p1s, // POINT PROJECTIONS into the first camera
    const std::vector<Eigen::Vector2d> &p2s, // POINT PROJECTIONS into the second camera
    const std::vector<Eigen::Vector2d> &p3s, // POINT PROJECTIONS into the third camera
    const std::vector<Eigen::Vector2d> &p4s, // POINT PROJECTIONS into the fourth camera
    const TrackSettings settings,           // track settings (loaded with load_settings)
    std::vector<Eigen::Matrix<double, 2, 4>>
        &P1_out, // OUTPUT: vector containing CAMERAS P1 that are consistent with the input + generate configurations
                 // with all points in front of the camera
    std::vector<Eigen::Matrix<double, 2, 4>>
        &P2_out, // OUTPUT: vector containing CAMERAS P2 that are consistent with the input + generate configurations
                 // with all points in front of the camera
    std::vector<Eigen::Matrix<double, 2, 4>>
        &P3_out, // OUTPUT: vector containing CAMERAS P3 that are consistent with the input + generate configurations
                 // with all points in front of the camera
    std::vector<Eigen::Matrix<double, 2, 4>>
        &P4_out, // OUTPUT: vector containing CAMERAS P4 that are consistent with the input + generate configurations
                 // with all points in front of the camera
    std::vector<int> &cam2QF, // OUTPUT: vector containing the ID of the quadrifocal vector for every camera quadruplet
    std::vector<Eigen::Matrix<double, 16, 1>>
        &QFs, // OUTPUT: vector containing 28 quadrifocal vectors consistent with the output
    std::vector<std::vector<Eigen::Vector3d>>
        &Xs); // OUTPUT: vector containing triangulated 3D points for every output camera quadruplet


// loads the track settings from a file (usually called trackParam.txt)
// returns true if the loading has succeeded, false otherwise
bool load_settings(std::string set_file,             // location of the file with the settings
                   struct TrackSettings &settings); // OUTPUT: variable where the settings is stored

// loads the starting problem + its solutions from a file (usually called starting_system.txt)
// returns true if the loading has succeeded, false otherwise
bool load_start_system(
    std::string data_file,                                 // location of the file with the starting problem
    std::vector<std::complex<double>> &problem,            // OUTPUT: coefficients of the starting problem
    std::vector<std::vector<std::complex<double>>> &sols); // OUTPUT: 28 solutions to the starting problem


}

#endif