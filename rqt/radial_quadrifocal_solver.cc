#include "radial_quadrifocal_solver.h"
#include "homotopy.h"
#include "solver_det4.h"

#include <Eigen/Core>
#include <Eigen/Dense>
#include <chrono>
#include <complex>
#include <float.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <math.h>
#include <string>
#include <vector>

namespace rqt {

#define Float double
typedef unsigned char ind;
static constexpr Float tol = 1e-3;
typedef std::complex<Float> complex;
using namespace std::chrono;
// using namespace std;

// FUNCTIONS

// loads the track settings
bool load_settings(std::string set_file, struct TrackSettings &settings) {
    std::ifstream f;
    f.open(set_file);

    if (!f.good()) {
        f.close();
        std::cout << "Settings file not available\n";

        return 0;
    }

    std::string t;

    // init dt
    double init_dt_;
    f >> init_dt_;
    getline(f, t);

    // min dt
    double min_dt_;
    f >> min_dt_;
    getline(f, t);

    // end zone factor
    double end_zone;
    f >> end_zone;
    getline(f, t);

    // corrector tolerance epsilon
    double epsilon;
    f >> epsilon;
    getline(f, t);

    // step increase factor
    double increase_factor;
    f >> increase_factor;
    getline(f, t);

    // infinity threshold
    double inf_thr;
    f >> inf_thr;
    getline(f, t);

    // max corr steps
    unsigned max_corr;
    f >> max_corr;
    getline(f, t);

    // num successes before increase
    unsigned succ_bef_inc;
    f >> succ_bef_inc;
    getline(f, t);

    f.close();
    settings.init_dt_ = init_dt_;
    settings.min_dt_ = min_dt_;
    settings.end_zone_factor_ = end_zone;
    settings.epsilon_ = epsilon;
    settings.epsilon2_ = epsilon * epsilon;
    settings.dt_increase_factor_ = increase_factor;
    settings.dt_decrease_factor_ = 1. / increase_factor;
    settings.infinity_threshold_ = inf_thr;
    settings.infinity_threshold2_ = inf_thr * inf_thr;
    settings.max_corr_steps_ = max_corr;
    settings.num_successes_before_increase_ = succ_bef_inc;

    return 1;
}

// loads the starting system
bool load_start_system(std::string data_file, std::vector<std::complex<double>> &problem,
                       std::vector<std::vector<std::complex<double>>> &sols) {
    std::ifstream f;
    f.open(data_file);

    if (!f.good()) {
        f.close();
        std::cout << "Data file not available\n";
        return 0;
    }

    problem = std::vector<std::complex<double>>(224);
    sols = std::vector<std::vector<std::complex<double>>>(56);

    // load the points
    for (int j = 0; j < 224; j++) {
        // skip the special characters
        char tr;
        f.get(tr);
        while (tr != '{' && tr != ',')
            f.get(tr);

        // load the real part
        double re;
        f >> re;

        // skip the special characters
        f.get(tr);
        while (tr != '{' && tr != ',')
            f.get(tr);

        // load the complex part
        double im;
        f >> im;

        std::complex<double> u = std::complex<double>(re, im);
        problem[j] = u;
    }

    // load the solutions
    for (int j = 0; j < 28; ++j) {
        std::vector<std::complex<double>> sol(13);
        for (int k = 0; k < 13; ++k) {
            // skip the special characters
            char tr;
            f.get(tr);
            while (tr != '{' && tr != ',')
                f.get(tr);

            // load the real part
            double re;
            f >> re;

            // skip the special characters
            f.get(tr);
            while (tr != '{' && tr != ',')
                f.get(tr);

            // load the complex part
            double im;
            f >> im;

            std::complex<double> u = std::complex<double>(re, im);

            sol[k] = u;
        }
        sols[j] = sol;
    }

    f.close();
    return 1;
}

// converts the solution to uncalibrated cameras
void sol2cam(std::complex<double> *solution, Eigen::Matrix<double, 2, 4> *Ps) {
    Eigen::Matrix<double, 2, 4> P1 = Eigen::Matrix<double, 2, 4>::Zero();
    P1(0, 0) = 1;
    P1(1, 0) = solution[0].real();
    P1(1, 1) = solution[0].real();
    P1(1, 2) = solution[0].real();
    P1(1, 3) = solution[0].real();
    Ps[0] = P1;

    Eigen::Matrix<double, 2, 4> P2 = Eigen::Matrix<double, 2, 4>::Zero();
    P2(0, 1) = 1;
    P2(1, 0) = solution[1].real();
    P2(1, 1) = solution[2].real();
    P2(1, 2) = solution[3].real();
    P2(1, 3) = solution[4].real();
    Ps[1] = P2;

    Eigen::Matrix<double, 2, 4> P3 = Eigen::Matrix<double, 2, 4>::Zero();
    P3(0, 2) = 1;
    P3(1, 0) = solution[5].real();
    P3(1, 1) = solution[6].real();
    P3(1, 2) = solution[7].real();
    P3(1, 3) = solution[8].real();
    Ps[2] = P3;

    Eigen::Matrix<double, 2, 4> P4 = Eigen::Matrix<double, 2, 4>::Zero();
    P4(0, 3) = 1;
    P4(1, 0) = solution[9].real();
    P4(1, 1) = solution[10].real();
    P4(1, 2) = solution[11].real();
    P4(1, 3) = solution[12].real();
    Ps[3] = P4;
}

// creates a symmetric system of cameras from the original system
void get_symmetric_cams(Eigen::Matrix<double, 2, 4> *Ps, Eigen::Matrix<double, 2, 4> *Ps_sym) {
    Eigen::Matrix<double, 2, 4> P1 = Ps[0];
    Eigen::Matrix<double, 2, 4> P2 = Ps[1];
    Eigen::Matrix<double, 2, 4> P3 = Ps[2];
    Eigen::Matrix<double, 2, 4> P4 = Ps[3];

    Eigen::Matrix4d odd_rows;
    odd_rows.row(0) = P1.row(1);
    odd_rows.row(1) = P2.row(1);
    odd_rows.row(2) = P3.row(1);
    odd_rows.row(3) = P4.row(1);

    Eigen::Matrix<double, 8, 4> sym_2 = Eigen::Matrix<double, 8, 4>::Zero();
    sym_2(0, 0) = 1;
    sym_2.row(1) = odd_rows.col(0).transpose();
    sym_2(2, 1) = 1;
    sym_2.row(3) = odd_rows.col(1).transpose();
    sym_2(4, 2) = 1;
    sym_2.row(5) = odd_rows.col(2).transpose();
    sym_2(6, 3) = 1;
    sym_2.row(7) = odd_rows.col(3).transpose();

    Eigen::Matrix<double, 8, 8> sym_1 = Eigen::Matrix<double, 8, 8>::Identity();
    sym_1(2, 2) = P2(1, 0) / P1(1, 0);
    sym_1(3, 3) = P2(1, 0) / P1(1, 0);
    sym_1(4, 4) = P3(1, 0) / P1(1, 0);
    sym_1(5, 5) = P3(1, 0) / P1(1, 0);
    sym_1(6, 6) = P4(1, 0) / P1(1, 0);
    sym_1(7, 7) = P4(1, 0) / P1(1, 0);

    Eigen::Matrix<double, 4, 4> sym_3 = Eigen::Matrix<double, 4, 4>::Identity();
    sym_3(1, 1) = P1(1, 0) / P2(1, 0);
    sym_3(2, 2) = P1(1, 0) / P3(1, 0);
    sym_3(3, 3) = P1(1, 0) / P4(1, 0);

    Eigen::Matrix<double, 8, 4> sym = sym_1 * sym_2 * sym_3;

    Eigen::Matrix<double, 2, 4> P1s = sym.block<2, 4>(0, 0);
    Ps_sym[0] = P1s;
    Eigen::Matrix<double, 2, 4> P2s = sym.block<2, 4>(2, 0);
    Ps_sym[1] = P2s;
    Eigen::Matrix<double, 2, 4> P3s = sym.block<2, 4>(4, 0);
    Ps_sym[2] = P3s;
    Eigen::Matrix<double, 2, 4> P4s = sym.block<2, 4>(6, 0);
    Ps_sym[3] = P4s;
}

// finds a quadrifocal tensor from cameras
Eigen::Matrix<double, 16, 1> cams2qft(Eigen::Matrix<double, 2, 4> *Ps) {
    Eigen::Matrix<double, 16, 1> QF;
    for (int a = 0; a < 16; ++a) {
        int ix4 = a % 2;
        int ix3 = (a >> 1) % 2;
        int ix2 = (a >> 2) % 2;
        int ix1 = (a >> 3) % 2;

        // det4(stackedCameras^{i,2+j,4+k,6+l}_{0,1,2,3}))
        Eigen::Matrix4d M;
        M.row(0) = Ps[0].row(ix1);
        M.row(1) = Ps[1].row(ix2);
        M.row(2) = Ps[2].row(ix3);
        M.row(3) = Ps[3].row(ix4);
        int sign = (ix1 + ix2 + ix3 + ix4) % 2;
        double c = M.determinant();
        if (sign)
            c = -1 * c;
        QF(a) = c;
    }
    QF = QF / QF.norm();
    return QF;
}

// for uncalibrated cameras finds all calibrated camera quadruplets with the same quadrifocal tensor + a frame fixed to
// P1 = [1 0 0 0; 0 1 0 0], P2 = [R1 R2 R3 1; R4 R5 R6 0]
int calibrate(Eigen::Matrix<double, 2, 4> *Ps, Eigen::Matrix<double, 2, 4> *P1cs, Eigen::Matrix<double, 2, 4> *P2cs,
              Eigen::Matrix<double, 2, 4> *P3cs, Eigen::Matrix<double, 2, 4> *P4cs, bool optimize_roots) {
    // calibrate the obtained cameras P1, P2, P3, P4

    // find the linear solutions to the system Pi^T*Q*Pi ~ I, Pi^T*Q*Pj = 0 => 2D kernel with basis Q1, Q2
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

    // find the element q1+alpha*q2 from the kernel that corresponds to a singluar matrix => solve for the feasible
    // alphas
    std::vector<double> alphas = solve_det4(Q1, Q2);
    // Eigen::VectorXd data = Eigen::VectorXd(20);
    // data.block<10,1>(0,0) = q1;
    // data.block<10,1>(10,0) = q2;
    // MatrixXcd alphas = solver_det4(data);

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

        // check if the matrix is positive semidefinite (it should be as it stems from H^T*H)
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

        // reorder the eigenvalues and eigenvectors such that the last one is the zero
        Eigen::Vector4d ev0 = V.col(zero_pos);
        double val0 = E(zero_pos);
        V.col(zero_pos) = V.col(3);
        E(zero_pos) = E(3);
        V.col(3) = ev0;
        E(3) = 1; // the last column of H has to be the last eigenvector, the trifocal tensor is valid for every
                  // multiple of it

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
        if (R1.determinant() < 0)
            R1.block<1, 3>(2, 0) = -r13.transpose();

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

        std::cout << "Calibration result -- res = " << (R1f * R1f.transpose() - Eigen::Matrix2d::Identity()).norm() << 
                     ", " << (R2f * R2f.transpose() - Eigen::Matrix2d::Identity()).norm() << 
                     ", " << (R3f * R3f.transpose() - Eigen::Matrix2d::Identity()).norm() << 
                     ", " << (R3f * R3f.transpose() - Eigen::Matrix2d::Identity()).norm() << std::endl;
                      

        P1cs[num_valid_sols] = P1f;
        P2cs[num_valid_sols] = P2f;
        P3cs[num_valid_sols] = P3f;
        P4cs[num_valid_sols] = P4f;
        ++num_valid_sols;
    }
    return num_valid_sols;
}

// for every given camera quadruplet returns all 16 camera quadruplets obtained by sign flips, which retain the
// quadrifocal tensor
int flip_cameras(const int num_calibrated, Eigen::Matrix<double, 2, 4> *P1cs, Eigen::Matrix<double, 2, 4> *P2cs,
                 Eigen::Matrix<double, 2, 4> *P3cs, Eigen::Matrix<double, 2, 4> *P4cs,
                 Eigen::Matrix<double, 2, 4> *P1fs, Eigen::Matrix<double, 2, 4> *P2fs,
                 Eigen::Matrix<double, 2, 4> *P3fs, Eigen::Matrix<double, 2, 4> *P4fs) {
    int num_flipped = 0;

    for (int a = 0; a < num_calibrated; ++a) {
        Eigen::Matrix<double, 2, 4> P1f = P1cs[a];
        Eigen::Matrix<double, 2, 4> P2f = P2cs[a];
        Eigen::Matrix<double, 2, 4> P3f = P3cs[a];
        Eigen::Matrix<double, 2, 4> P4f = P4cs[a];

        // generate all possible cameras by flipping the signs
        // original camera
        P1fs[16 * a + 0] = P1f;
        P2fs[16 * a + 0] = P2f;
        P3fs[16 * a + 0] = P3f;
        P4fs[16 * a + 0] = P4f;

        // flipped 3rd column
        P1fs[16 * a + 1] = P1f;
        P1fs[16 * a + 1].col(2) = -P1f.col(2);
        P2fs[16 * a + 1] = P2f;
        P2fs[16 * a + 1].col(2) = -P2f.col(2);
        P3fs[16 * a + 1] = P3f;
        P3fs[16 * a + 1].col(2) = -P3f.col(2);
        P4fs[16 * a + 1] = P4f;
        P4fs[16 * a + 1].col(2) = -P4f.col(2);

        // flipped "2nd camera"
        P1fs[16 * a + 2] = P1f;
        P2fs[16 * a + 2] = -P2f;
        P3fs[16 * a + 2] = P3f;
        P4fs[16 * a + 2] = P4f;
        P2fs[16 * a + 2].col(3) = P2f.col(3);
        P3fs[16 * a + 2].col(3) = -P3f.col(3);
        P4fs[16 * a + 2].col(3) = -P4f.col(3);

        // flipped 3rd column + "2nd camera"
        P1fs[16 * a + 3] = P1fs[16 * a + 1];
        P2fs[16 * a + 3] = -P2fs[16 * a + 1];
        P3fs[16 * a + 3] = P3fs[16 * a + 1];
        P4fs[16 * a + 3] = P4fs[16 * a + 1];
        P2fs[16 * a + 3].col(3) = P2fs[16 * a + 1].col(3);
        P3fs[16 * a + 3].col(3) = -P3fs[16 * a + 1].col(3);
        P4fs[16 * a + 3].col(3) = -P4fs[16 * a + 1].col(3);

        // flipped 3rd camera
        P1fs[16 * a + 4] = P1fs[16 * a + 0];
        P2fs[16 * a + 4] = P2fs[16 * a + 0];
        P3fs[16 * a + 4] = -P3fs[16 * a + 0];
        P4fs[16 * a + 4] = P4fs[16 * a + 0];

        // flipped 3rd column + 3rd camera
        P1fs[16 * a + 5] = P1fs[16 * a + 1];
        P2fs[16 * a + 5] = P2fs[16 * a + 1];
        P3fs[16 * a + 5] = -P3fs[16 * a + 1];
        P4fs[16 * a + 5] = P4fs[16 * a + 1];

        // flipped "2nd camera" + 3rd camera
        P1fs[16 * a + 6] = P1fs[16 * a + 2];
        P2fs[16 * a + 6] = P2fs[16 * a + 2];
        P3fs[16 * a + 6] = -P3fs[16 * a + 2];
        P4fs[16 * a + 6] = P4fs[16 * a + 2];

        // flipped 3rd column + "2nd camera" + 3rd camera
        P1fs[16 * a + 7] = P1fs[16 * a + 3];
        P2fs[16 * a + 7] = P2fs[16 * a + 3];
        P3fs[16 * a + 7] = -P3fs[16 * a + 3];
        P4fs[16 * a + 7] = P4fs[16 * a + 3];

        // flipped 4th camera
        P1fs[16 * a + 8] = P1fs[16 * a + 0];
        P2fs[16 * a + 8] = P2fs[16 * a + 0];
        P3fs[16 * a + 8] = P3fs[16 * a + 0];
        P4fs[16 * a + 8] = -P4fs[16 * a + 0];

        // flipped 3rd column + 4th camera
        P1fs[16 * a + 9] = P1fs[16 * a + 1];
        P2fs[16 * a + 9] = P2fs[16 * a + 1];
        P3fs[16 * a + 9] = P3fs[16 * a + 1];
        P4fs[16 * a + 9] = -P4fs[16 * a + 1];

        // flipped "2nd camera" + 4th camera
        P1fs[16 * a + 10] = P1fs[16 * a + 2];
        P2fs[16 * a + 10] = P2fs[16 * a + 2];
        P3fs[16 * a + 10] = P3fs[16 * a + 2];
        P4fs[16 * a + 10] = -P4fs[16 * a + 2];

        // flipped 3rd column + "2nd camera" + 4th camera
        P1fs[16 * a + 11] = P1fs[16 * a + 3];
        P2fs[16 * a + 11] = P2fs[16 * a + 3];
        P3fs[16 * a + 11] = P3fs[16 * a + 3];
        P4fs[16 * a + 11] = -P4fs[16 * a + 3];

        // flipped 3rd camera + 4th camera
        P1fs[16 * a + 12] = P1fs[16 * a + 0];
        P2fs[16 * a + 12] = P2fs[16 * a + 0];
        P3fs[16 * a + 12] = -P3fs[16 * a + 0];
        P4fs[16 * a + 12] = -P4fs[16 * a + 0];

        // flipped 3rd column + 3rd camera + 4th camera
        P1fs[16 * a + 13] = P1fs[16 * a + 1];
        P2fs[16 * a + 13] = P2fs[16 * a + 1];
        P3fs[16 * a + 13] = -P3fs[16 * a + 1];
        P4fs[16 * a + 13] = -P4fs[16 * a + 1];

        // flipped "2nd camera" + 3rd camera + 4th camera
        P1fs[16 * a + 14] = P1fs[16 * a + 2];
        P2fs[16 * a + 14] = P2fs[16 * a + 2];
        P3fs[16 * a + 14] = -P3fs[16 * a + 2];
        P4fs[16 * a + 14] = -P4fs[16 * a + 2];

        // flipped 3rd column + "2nd camera" + 3rd camera + 4th camera
        P1fs[16 * a + 15] = P1fs[16 * a + 3];
        P2fs[16 * a + 15] = P2fs[16 * a + 3];
        P3fs[16 * a + 15] = -P3fs[16 * a + 3];
        P4fs[16 * a + 15] = -P4fs[16 * a + 3];

        num_flipped += 16;
    }

    return num_flipped;
}

// for a given camera quadruplet + point projections finds triangulated 3D points; returns true iff the configuration
// allows all points in front of all cameras
bool triangulate(const std::vector<Eigen::Vector2d> &p1s, const std::vector<Eigen::Vector2d> &p2s,
                 const std::vector<Eigen::Vector2d> &p3s, const std::vector<Eigen::Vector2d> &p4s,
                 Eigen::Matrix<double, 2, 4> P1, Eigen::Matrix<double, 2, 4> P2, Eigen::Matrix<double, 2, 4> P3,
                 Eigen::Matrix<double, 2, 4> P4, std::vector<Eigen::Vector3d> &Xs) {
    // count the number of cases where we want to flip the camera vs. the cases where we want to keep it
    // if the final number is not in {-13, 13}, skip the camera, otherwise either store it or flip and store
    int flip1 = 0;
    int flip2 = 0;
    int flip3 = 0;
    int flip4 = 0;

    Eigen::Matrix2d eps2;
    eps2 << 0, 1, -1, 0;

    // triangulate every 3D point
    for (int a = 0; a < 13; ++a) {
        // obtain the system whose solution is the homogeneous 3D point
        Eigen::Matrix4d A;
        A.row(0) = p1s[a].transpose() * eps2 * P1;
        A.row(1) = p2s[a].transpose() * eps2 * P2;
        A.row(2) = p3s[a].transpose() * eps2 * P3;
        A.row(3) = p4s[a].transpose() * eps2 * P4;

        // use SVD to obtain the kernel of the matrix
        Eigen::JacobiSVD<Eigen::Matrix4d> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Matrix4d V = svd.matrixV();
        Eigen::Vector4d X = V.col(3);
        X = X / X(3);
        Xs[a] = X.block<3, 1>(0, 0);

        // check if the X is on the good side of the problem
        // get the projections by the camera
        Eigen::Vector2d pr1 = P1 * X;
        Eigen::Vector2d pr2 = P2 * X;
        Eigen::Vector2d pr3 = P3 * X;
        Eigen::Vector2d pr4 = P4 * X;

        // check if the projection is on the same side as the original projection or it is flipped
        if (p1s[a](0) * pr1(0) > 0)
            ++flip1;
        else
            --flip1;

        if (p2s[a](0) * pr2(0) > 0)
            ++flip2;
        else
            --flip2;

        if (p3s[a](0) * pr3(0) > 0)
            ++flip3;
        else
            --flip3;

        if (p4s[a](0) * pr4(0) > 0)
            ++flip4;
        else
            --flip4;
    }

    if (flip1 == -13) {
        flip1 = 13;
        // P1 = -P1;
    }
    if (flip2 == -13) {
        flip2 = 13;
        // P2 = -P2;
    }
    if (flip3 == -13) {
        flip3 = 13;
        // P3 = -P3;
    }
    if (flip4 == -13) {
        flip4 = 13;
        // P4 = -P4;
    }

    // only accept the pose if all points are in front of all cameras
    if (flip1 == 13 && flip2 == 13 && flip3 == 13 && flip4 == 13)
        return true;
    return false;
}

// solves one instance of the radial quadrifocal problem
int radial_quadrifocal_solver(const std::vector<std::complex<double>> start_problem,
                              const std::vector<std::vector<std::complex<double>>> start_sols,
                              const std::vector<Eigen::Vector2d> &p1s, const std::vector<Eigen::Vector2d> &p2s,
                              const std::vector<Eigen::Vector2d> &p3s, const std::vector<Eigen::Vector2d> &p4s,
                              const TrackSettings settings,
                              std::vector<Eigen::Matrix<double, 2, 4>> &P1_out,
                              std::vector<Eigen::Matrix<double, 2, 4>> &P2_out,
                              std::vector<Eigen::Matrix<double, 2, 4>> &P3_out,
                              std::vector<Eigen::Matrix<double, 2, 4>> &P4_out,
                              std::vector<Eigen::Matrix<double, 16, 1>> &QFs) {
    // initialize the variables for the tracking
    std::complex<double> params[448];
    static std::complex<double> solution[13];
    int num_steps;

    int num_good_poses = 0;

    // copy the start problem
    for (int a = 0; a < 224; a++) {
        params[a] = start_problem[a];
    }

    // convert the 2D points into the coefficients used by the solver
    Eigen::Matrix<double, 13, 16> coef_matrix;
    Eigen::Matrix2d eps;
    eps << 0, 1, 1, 0;
    for (int a = 0; a < 13; ++a) {
        // obtain lines on which the point projections lie
        Eigen::Vector2d l1 = eps * p1s[a];
        l1 = l1 / l1.norm();
        Eigen::Vector2d l2 = eps * p2s[a];
        l2 = l2 / l2.norm();
        Eigen::Vector2d l3 = eps * p3s[a];
        l3 = l3 / l3.norm();
        Eigen::Vector2d l4 = eps * p4s[a];
        l4 = l4 / l4.norm();

        // generate the coefficients of the problem + store them to the tracking parameters
        for (int b = 0; b < 16; ++b) {
            int ix4 = b % 2;
            int ix3 = (b >> 1) % 2;
            int ix2 = (b >> 2) % 2;
            int ix1 = (b >> 3) % 2;

            double c = l1(ix1) * l2(ix2) * l3(ix3) * l4(ix4);
            params[224 + 16 * a + b] = c;
            coef_matrix(a, b) = c;
        }
    }
    // if the system is badly conditioned, do not compute it
    Eigen::JacobiSVD<Eigen::Matrix<double, 13, 16>> svd(coef_matrix, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix<double, 13, 1> svs = svd.singularValues();
    double cond = svs(12);
    if (cond < 1e-10)
        return 0;

    
    // track from every solution to the starting system
    for (int k = 0; k < 28; ++k) {
        // copy the solution
        std::complex<double> cur_start[13];
        for (int a = 0; a < 13; ++a) {
            cur_start[a] = start_sols[k][a];
        }

        // track the problem
        int status = track(settings, cur_start, params, solution, &num_steps);

        // convert the solutions into cameras and triangulate the 3D points
        if (status == 2) {
            // build the cameras from the IDs
            Eigen::Matrix<double, 2, 4> Ps[4];
            sol2cam(solution, Ps);

            const Eigen::Matrix<double, 16, 1> QF = cams2qft(Ps);
            

            P1_out.push_back(Ps[0]);
            P2_out.push_back(Ps[1]);
            P3_out.push_back(Ps[2]);
            P4_out.push_back(Ps[3]);
            QFs.push_back(QF);

            // get the symmetric cameras
            Eigen::Matrix<double, 2, 4> Ps_sym[4];
            get_symmetric_cams(Ps, Ps_sym);

            P1_out.push_back(Ps_sym[0]);
            P2_out.push_back(Ps_sym[1]);
            P3_out.push_back(Ps_sym[2]);
            P4_out.push_back(Ps_sym[3]);
            QFs.push_back(QF);
        }
    }
    return P1_out.size();
}

} // namespace rqt