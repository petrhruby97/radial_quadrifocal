#include "metric_upgrade.h"

#include "solver_det4.h"

namespace rqt {

int filter_cheirality(const std::vector<Eigen::Vector2d> &x1, const std::vector<Eigen::Vector2d> &x2,
                      const std::vector<Eigen::Vector2d> &x3, const std::vector<Eigen::Vector2d> &x4,
                      const Eigen::Matrix<double, 2, 4> &P1, const Eigen::Matrix<double, 2, 4> &P2,
                      const Eigen::Matrix<double, 2, 4> &P3, const Eigen::Matrix<double, 2, 4> &P4,
                      std::vector<Eigen::Matrix<double, 2, 4>> &P1_calib,
                      std::vector<Eigen::Matrix<double, 2, 4>> &P2_calib,
                      std::vector<Eigen::Matrix<double, 2, 4>> &P3_calib,
                      std::vector<Eigen::Matrix<double, 2, 4>> &P4_calib,
                      std::vector<std::vector<Eigen::Vector3d>> &Xs) {
    Eigen::Matrix2d eps2;
    eps2 << 0, 1, -1, 0;
    const int sample_sz = x1.size();
    std::vector<Eigen::Vector3d> X(sample_sz);
    double s1, s2, s3, s4;
    for (int k = 0; k < sample_sz; ++k) {
        // obtain the system whose solution is the homogeneous 3D point
        Eigen::Matrix4d A;
        A.row(0) = x1[k].transpose() * eps2 * P1;
        A.row(1) = x2[k].transpose() * eps2 * P2;
        A.row(2) = x3[k].transpose() * eps2 * P3;
        A.row(3) = x4[k].transpose() * eps2 * P4;

        // use SVD to obtain the kernel of the matrix
        Eigen::JacobiSVD<Eigen::Matrix4d> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Matrix4d V = svd.matrixV();
        X[k] = V.col(3).hnormalized();

        Eigen::Vector2d pr1 = P1 * X[k].homogeneous();
        Eigen::Vector2d pr2 = P2 * X[k].homogeneous();
        Eigen::Vector2d pr3 = P3 * X[k].homogeneous();
        Eigen::Vector2d pr4 = P4 * X[k].homogeneous();

        const double alpha1 = pr1.dot(x1[k]);
        const double alpha2 = pr2.dot(x2[k]);
        const double alpha3 = pr3.dot(x3[k]);
        const double alpha4 = pr4.dot(x4[k]);

        if (k == 0) {
            s1 = alpha1 > 0 ? 1 : -1;
            s2 = alpha2 > 0 ? 1 : -1;
            s3 = alpha3 > 0 ? 1 : -1;
            s4 = alpha4 > 0 ? 1 : -1;
        } else {
            if (s1 * alpha1 < 0 || s2 * alpha2 < 0 || s3 * alpha3 < 0 || s4 * alpha4 < 0) {
                return 0;
            }
        }
    }

    Eigen::Matrix4d H;
    H.setIdentity();

    if (s1 < 0) {
        // ensures that P1 = [I_2, 0]
        H(0, 0) = H(1, 1) = -1;
        for (Eigen::Vector3d &Xk : X) {
            Xk.topRows<2>() *= -1;
        }
    }

    P1_calib.push_back(P1);
    P2_calib.push_back(s2 * P2 * H);
    P3_calib.push_back(s3 * P3 * H);
    P4_calib.push_back(s4 * P4 * H);
    Xs.push_back(X);

    return 1;
}

int metric_upgrade(const std::vector<Eigen::Vector2d> &x1, const std::vector<Eigen::Vector2d> &x2,
                   const std::vector<Eigen::Vector2d> &x3, const std::vector<Eigen::Vector2d> &x4,
                   const Eigen::Matrix<double, 2, 4> &P1, const Eigen::Matrix<double, 2, 4> &P2,
                   const Eigen::Matrix<double, 2, 4> &P3, const Eigen::Matrix<double, 2, 4> &P4,
                   std::vector<Eigen::Matrix<double, 2, 4>> &P1_calib,
                   std::vector<Eigen::Matrix<double, 2, 4>> &P2_calib,
                   std::vector<Eigen::Matrix<double, 2, 4>> &P3_calib,
                   std::vector<Eigen::Matrix<double, 2, 4>> &P4_calib, std::vector<std::vector<Eigen::Vector3d>> &Xs) {

    std::vector<Eigen::Matrix<double, 2, 4>> Ps = {P1, P2, P3, P4};
    Eigen::Matrix<double, 8, 10> M;
    for (int a = 0; a < 4; ++a) {
        M(2 * a, 0) = Ps[a](0, 0) * Ps[a](0, 0) - Ps[a](1, 0) * Ps[a](1, 0);
        M(2 * a, 1) = 2 * Ps[a](0, 0) * Ps[a](0, 1) - 2 * Ps[a](1, 0) * Ps[a](1, 1);
        M(2 * a, 2) = 2 * Ps[a](0, 0) * Ps[a](0, 2) - 2 * Ps[a](1, 0) * Ps[a](1, 2);
        M(2 * a, 3) = 2 * Ps[a](0, 0) * Ps[a](0, 3) - 2 * Ps[a](1, 0) * Ps[a](1, 3);
        M(2 * a, 4) = Ps[a](0, 1) * Ps[a](0, 1) - Ps[a](1, 1) * Ps[a](1, 1);
        M(2 * a, 5) = 2 * Ps[a](0, 1) * Ps[a](0, 2) - 2 * Ps[a](1, 1) * Ps[a](1, 2);
        M(2 * a, 6) = 2 * Ps[a](0, 1) * Ps[a](0, 3) - 2 * Ps[a](1, 1) * Ps[a](1, 3);
        M(2 * a, 7) = Ps[a](0, 2) * Ps[a](0, 2) - Ps[a](1, 2) * Ps[a](1, 2);
        M(2 * a, 8) = 2 * Ps[a](0, 2) * Ps[a](0, 3) - 2 * Ps[a](1, 2) * Ps[a](1, 3);
        M(2 * a, 9) = Ps[a](0, 3) * Ps[a](0, 3) - Ps[a](1, 3) * Ps[a](1, 3);

        M(2 * a + 1, 0) = Ps[a](0, 0) * Ps[a](1, 0);
        M(2 * a + 1, 1) = Ps[a](0, 0) * Ps[a](1, 1) + Ps[a](0, 1) * Ps[a](1, 0);
        M(2 * a + 1, 2) = Ps[a](0, 0) * Ps[a](1, 2) + Ps[a](0, 2) * Ps[a](1, 0);
        M(2 * a + 1, 3) = Ps[a](0, 0) * Ps[a](1, 3) + Ps[a](0, 3) * Ps[a](1, 0);
        M(2 * a + 1, 4) = Ps[a](0, 1) * Ps[a](1, 1);
        M(2 * a + 1, 5) = Ps[a](0, 1) * Ps[a](1, 2) + Ps[a](0, 2) * Ps[a](1, 1);
        M(2 * a + 1, 6) = Ps[a](0, 1) * Ps[a](1, 3) + Ps[a](0, 3) * Ps[a](1, 1);
        M(2 * a + 1, 7) = Ps[a](0, 2) * Ps[a](1, 2);
        M(2 * a + 1, 8) = Ps[a](0, 2) * Ps[a](1, 3) + Ps[a](0, 3) * Ps[a](1, 2);
        M(2 * a + 1, 9) = Ps[a](0, 3) * Ps[a](1, 3);
    }
    Eigen::JacobiSVD<Eigen::Matrix<double, 8, 10>> svd(M, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix<double, 10, 1> q1 = svd.matrixV().col(8);
    Eigen::Matrix<double, 10, 1> q2 = svd.matrixV().col(9);

    // change the kernel basis q1, q2 into a matrix form
    Eigen::Matrix4d Q1;
    Q1(0, 0) = q1(0);
    Q1(0, 1) = q1(1);
    Q1(0, 2) = q1(2);
    Q1(0, 3) = q1(3);
    Q1(1, 0) = q1(1);
    Q1(1, 1) = q1(4);
    Q1(1, 2) = q1(5);
    Q1(1, 3) = q1(6);
    Q1(2, 0) = q1(2);
    Q1(2, 1) = q1(5);
    Q1(2, 2) = q1(7);
    Q1(2, 3) = q1(8);
    Q1(3, 0) = q1(3);
    Q1(3, 1) = q1(6);
    Q1(3, 2) = q1(8);
    Q1(3, 3) = q1(9);

    Eigen::Matrix4d Q2;
    Q2(0, 0) = q2(0);
    Q2(0, 1) = q2(1);
    Q2(0, 2) = q2(2);
    Q2(0, 3) = q2(3);
    Q2(1, 0) = q2(1);
    Q2(1, 1) = q2(4);
    Q2(1, 2) = q2(5);
    Q2(1, 3) = q2(6);
    Q2(2, 0) = q2(2);
    Q2(2, 1) = q2(5);
    Q2(2, 2) = q2(7);
    Q2(2, 3) = q2(8);
    Q2(3, 0) = q2(3);
    Q2(3, 1) = q2(6);
    Q2(3, 2) = q2(8);
    Q2(3, 3) = q2(9);

    // find the element q1+alpha*q2 from the kernel that corresponds to a singluar
    // matrix => solve for the feasible alphas
    std::vector<double> alphas = solve_det4(Q1, Q2);

    // for every alpha obtain the calibrated camera matrices
    int num_valid_sols = 0;
    for (int q = 0; q < alphas.size(); ++q) {
        double alpha = alphas[q];

        // get the matrix Q
        // Eigen::Matrix4d Q = Q1+alphas(q).real()*Q2;
        Eigen::Matrix4d Q = Q1 + alpha * Q2;

        // check the validity of Q
        if (std::abs(Q.determinant()) > 1e-6)
            continue;

        // get the eigenvalues
        Eigen::EigenSolver<Eigen::Matrix4d> es(Q);
        Eigen::Matrix4d V = es.eigenvectors().real();
        Eigen::Vector4d E = es.eigenvalues().real();

        // check if the matrix is positive semidefinite (it should be as it stems
        // from H^T*H)
        if (es.eigenvalues().real().maxCoeff() < 1e-6)
            E = -E;
        else if (es.eigenvalues().real().minCoeff() < -1e-6)
            continue;

        // make the "soft zero" value nonnegative (so that we can perform sqrt)
        int zero_pos = -1;
        double min_ev = 9999;
        for (int qq = 0; qq < 4; ++qq) {
            if (E(qq) < 0)
                E(qq) = -E(qq);

            if (E(qq) < min_ev) {
                min_ev = E(qq);
                zero_pos = qq;
            }
        }

        // reorder the eigenvalues and eigenvectors such that the last one is the
        // zero
        Eigen::Vector4d ev0 = V.col(zero_pos);
        double val0 = E(zero_pos);
        V.col(zero_pos) = V.col(3);
        E(zero_pos) = E(3);
        V.col(3) = ev0;
        E(3) = 1; // the last column of H has to be the last eigenvector, the
                  // trifocal tensor is valid for every multiple of it

        // find the transformation matrix H as the "square root" of the matrix Q
        Eigen::Matrix4d H = V * E.cwiseSqrt().asDiagonal();

        // transform the cameras by H as Pc = P*H
        Eigen::Matrix<double, 2, 4> P1c = Ps[0] * H;
        Eigen::Matrix<double, 2, 4> P2c = Ps[1] * H;
        Eigen::Matrix<double, 2, 4> P3c = Ps[2] * H;
        Eigen::Matrix<double, 2, 4> P4c = Ps[3] * H;

        // normalize the cameras
        P1c = P1c / P1c.block<1, 3>(0, 0).norm();
        P2c = P2c / P2c.block<1, 3>(0, 0).norm();
        P3c = P3c / P3c.block<1, 3>(0, 0).norm();
        P4c = P4c / P4c.block<1, 3>(0, 0).norm();

        // extract the rotation matrices
        Eigen::Vector3d r13 = P1c.block<1, 3>(0, 0).transpose().cross(P1c.block<1, 3>(1, 0).transpose());
        r13 = r13 / r13.norm();
        Eigen::Matrix3d R1;
        R1.block<2, 3>(0, 0) = P1c.block<2, 3>(0, 0);
        R1.block<1, 3>(2, 0) = r13.transpose();

        // fix the frame
        Eigen::Matrix<double, 2, 3> R1f = P1c.block<2, 3>(0, 0) * R1.transpose();
        Eigen::Matrix<double, 2, 3> R2f = P2c.block<2, 3>(0, 0) * R1.transpose();
        Eigen::Matrix<double, 2, 3> R3f = P3c.block<2, 3>(0, 0) * R1.transpose();
        Eigen::Matrix<double, 2, 3> R4f = P4c.block<2, 3>(0, 0) * R1.transpose();
        double t12 = (P2c(1, 3) - R2f(1, 0) * P1c(0, 3) - R2f(1, 1) * P1c(1, 3)) / R2f(1, 2);
        Eigen::Vector3d t1(P1c(0, 3), P1c(1, 3), t12);
        Eigen::Vector2d t1f = P1c.col(3) - R1f * t1;
        Eigen::Vector2d t2f = P2c.col(3) - R2f * t1;
        Eigen::Vector2d t3f = P3c.col(3) - R3f * t1;
        Eigen::Vector2d t4f = P4c.col(3) - R4f * t1;

        double sc = t2f(0);
        t1f = t1f / sc;
        t2f = t2f / sc;
        t3f = t3f / sc;
        t4f = t4f / sc;

        // finalize the calibrated and fixed cameras
        Eigen::Matrix<double, 2, 4> P1f;
        P1f.block<2, 3>(0, 0) = R1f;
        P1f.col(3) = t1f;

        Eigen::Matrix<double, 2, 4> P2f;
        P2f.block<2, 3>(0, 0) = R2f;
        P2f.col(3) = t2f;

        Eigen::Matrix<double, 2, 4> P3f;
        P3f.block<2, 3>(0, 0) = R3f;
        P3f.col(3) = t3f;

        Eigen::Matrix<double, 2, 4> P4f;
        P4f.block<2, 3>(0, 0) = R4f;
        P4f.col(3) = t4f;

        /*
        std::cout << "Calibration result -- res = " << (R1f * R1f.transpose() -
        Eigen::Matrix2d::Identity()).norm() <<
                     ", " << (R2f * R2f.transpose() -
        Eigen::Matrix2d::Identity()).norm() <<
                     ", " << (R3f * R3f.transpose() -
        Eigen::Matrix2d::Identity()).norm() <<
                     ", " << (R3f * R3f.transpose() -
        Eigen::Matrix2d::Identity()).norm() << std::endl;
        */

        int valid = filter_cheirality(x1, x2, x3, x4, P1f, P2f, P3f, P4f, P1_calib, P2_calib, P3_calib, P4_calib, Xs);
        num_valid_sols += valid;

        /*
        // TODO Figure out if we need this?
        // Check -w flip
        P2f.col(3) *= -1;
        P3f.col(3) *= -1;
        P4f.col(3) *= -1;
        valid = filter_cheirality(x1,x2,x3,x4,P1f, P2f, P3f, P4f, P1_calib,
        P2_calib, P3_calib, P4_calib, Xs); num_valid_sols += valid;
        */
    }
    return num_valid_sols;

    return 0;
}

} // namespace rqt
