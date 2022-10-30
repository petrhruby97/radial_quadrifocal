#ifndef RQT_SOLVER_L_
#define RQT_SOLVER_L_

#include "types.h"

#include <Eigen/Core>
#include <Eigen/Dense>

namespace rqt {

// Returns the projective cameras in P1_out, etc. Corresponding quadrifocal
// tensors in QFs
int linear_radial_quadrifocal_solver(const std::vector<Eigen::Vector2d> &p1s, const std::vector<Eigen::Vector2d> &p2s,
                              const std::vector<Eigen::Vector2d> &p3s, const std::vector<Eigen::Vector2d> &p4s,
                              const StartSystem &start_system, const TrackSettings &settings,
                              std::vector<Eigen::Matrix<double, 2, 4>> &P1_out,
                              std::vector<Eigen::Matrix<double, 2, 4>> &P2_out,
                              std::vector<Eigen::Matrix<double, 2, 4>> &P3_out,
                              std::vector<Eigen::Matrix<double, 2, 4>> &P4_out,
                              std::vector<Eigen::Matrix<double, 16, 1>> &QFs);

}
#endif
