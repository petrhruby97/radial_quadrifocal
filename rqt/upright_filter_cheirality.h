#ifndef RQT_FILTER_CHEIRALITY_U_
#define RQT_FILTER_CHEIRALITY_U_

#include <Eigen/Dense>

namespace rqt {

// for uncalibrated cameras finds all calibrated camera quadruplets with the
// same quadrifocal tensor + a frame fixed to P1 = [1 0 0 0; 0 1 0 0], P2 = [R1
// R2 R3 1; R4 R5 R6 0]
int upright_filter_cheirality(const std::vector<Eigen::Vector2d> &x1, const std::vector<Eigen::Vector2d> &x2,
                      const std::vector<Eigen::Vector2d> &x3, const std::vector<Eigen::Vector2d> &x4,
                      const Eigen::Matrix<double, 2, 4> &P1, const Eigen::Matrix<double, 2, 4> &P2,
                      const Eigen::Matrix<double, 2, 4> &P3, const Eigen::Matrix<double, 2, 4> &P4,
                      std::vector<Eigen::Matrix<double, 2, 4>> &P1_calib,
                      std::vector<Eigen::Matrix<double, 2, 4>> &P2_calib,
                      std::vector<Eigen::Matrix<double, 2, 4>> &P3_calib,
                      std::vector<Eigen::Matrix<double, 2, 4>> &P4_calib,
                      std::vector<std::vector<Eigen::Vector3d>> &Xs);

} // namespace rqt

#endif
