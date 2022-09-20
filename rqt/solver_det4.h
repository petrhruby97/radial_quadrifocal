#ifndef RQT_SOLVER_DET4
#define RQT_SOLVER_DET4

#include <Eigen/Dense>

namespace rqt {

// Finds alpha such that det(A + alpha*B) = 0
std::vector<double> solve_det4(const Eigen::Matrix4d &A,
                               const Eigen::Matrix4d &B);

} // namespace rqt

#endif