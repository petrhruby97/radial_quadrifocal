// \author Petr Hruby and Viktor Larsson
#include "radial_quadrifocal_solver.h"

#include "homotopy.h"
#include "solver_det4.h"

#include <Eigen/Core>
#include <Eigen/Dense>
#include <chrono>
#include <complex>
#include <float.h>
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

// solves one instance of the radial quadrifocal problem
int radial_quadrifocal_solver(const std::vector<Eigen::Vector2d> &p1s, const std::vector<Eigen::Vector2d> &p2s,
                              const std::vector<Eigen::Vector2d> &p3s, const std::vector<Eigen::Vector2d> &p4s,
                              const StartSystem &start_system, const TrackSettings &settings,
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
        params[a] = start_system.problem[a];
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

        // generate the coefficients of the problem + store them to the tracking
        // parameters
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
            cur_start[a] = start_system.sols[k][a];
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
