#ifndef RQT_SOLVER_N_
#define RQT_SOLVER_N_

#include "types.h"

#include <Eigen/Core>
#include <Eigen/Dense>

namespace rqt {

// Returns the projective cameras in P1_out, etc. Corresponding quadrifocal
// tensors in QFs
int nanson_radial_quadrifocal_solver(const std::vector<Eigen::Vector2d> &p1s, const std::vector<Eigen::Vector2d> &p2s,
                              const std::vector<Eigen::Vector2d> &p3s, const std::vector<Eigen::Vector2d> &p4s,
                              const StartSystem &start_system, const TrackSettings &settings,
                              std::vector<Eigen::Matrix<double, 2, 4>> &P1_out,
                              std::vector<Eigen::Matrix<double, 2, 4>> &P2_out,
                              std::vector<Eigen::Matrix<double, 2, 4>> &P3_out,
                              std::vector<Eigen::Matrix<double, 2, 4>> &P4_out,
                              std::vector<Eigen::Matrix<double, 16, 1>> &QFs);

// loads the track settings from a file (usually called trackParam.txt)
// returns true if the loading has succeeded, false otherwise
/*bool load_settings(std::string set_file,            // location of the file with the settings
                   struct TrackSettings &settings); // OUTPUT: variable where the settings is stored

// loads the starting problem + its solutions from a file (usually called
// starting_system.txt) returns true if the loading has succeeded, false
// otherwise
bool load_start_system(
    std::string data_file,                                 // location of the file with the starting problem
    std::vector<std::complex<double>> &problem,            // OUTPUT: coefficients of the starting problem
    std::vector<std::vector<std::complex<double>>> &sols); // OUTPUT: 28 solutions to the starting problem
*/

} // namespace rqt

#endif
