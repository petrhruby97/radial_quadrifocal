#include "upright_radial_quadrifocal_solver.h"

#include "upright_homotopy.h"

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

// converts the solution to calibrated cameras
void upright_sol2cam(std::complex<double> * solution, Eigen::Matrix<double,2,4> *Ps)
{

	Eigen::Matrix<double,2,4> P1 = Eigen::Matrix<double,2,4>::Identity();

        Eigen::Matrix<double,2,4> P2 = Eigen::Matrix<double,2,4>::Identity();
        const double div2 = 1+(solution[0].real()*solution[0].real());
        P2(0,0) = (1-solution[0].real()*solution[0].real())/div2;
        P2(0,2) = 2*solution[0].real()/div2;
        P2(1,3) = solution[7].real()/div2;

        Eigen::Matrix<double,2,4> P3 = Eigen::Matrix<double,2,4>::Identity();
        const double div3 = 1+(solution[1].real()*solution[1].real());
        P3(0,0) = (1-solution[1].real()*solution[1].real())/div3;
        P3(0,2) = 2*solution[1].real()/div3;
        P3(0,3) = solution[3].real()/div3;
        P3(1,3) = solution[4].real()/div3;

        Eigen::Matrix<double,2,4> P4 = Eigen::Matrix<double,2,4>::Identity();
        const double div4 = 1+(solution[2].real()*solution[2].real());
        P4(0,0) = (1-solution[2].real()*solution[2].real())/div4;
        P4(0,2) = 2*solution[2].real()/div4;
        P4(0,3) = solution[5].real()/div4;
        P4(1,3) = solution[6].real()/div4;

        double sc = P2.col(3).norm() + P3.col(3).norm() + P4.col(3).norm();
        P2.col(3) = P2.col(3)/sc;
        P3.col(3) = P3.col(3)/sc;
        P4.col(3) = P4.col(3)/sc;

	Ps[0] = P1;
	Ps[1] = P2;
	Ps[2] = P3;
	Ps[3] = P4;
}

// finds a quadrifocal tensor from cameras
Eigen::Matrix<double, 16, 1> upright_cams2qft(Eigen::Matrix<double, 2, 4> *Ps) {
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
int upright_radial_quadrifocal_solver(const std::vector<Eigen::Vector2d> &p1s, const std::vector<Eigen::Vector2d> &p2s,
                              const std::vector<Eigen::Vector2d> &p3s, const std::vector<Eigen::Vector2d> &p4s,
                              const StartSystem &start_system, const TrackSettings &settings,
                              std::vector<Eigen::Matrix<double, 2, 4>> &P1_out,
                              std::vector<Eigen::Matrix<double, 2, 4>> &P2_out,
                              std::vector<Eigen::Matrix<double, 2, 4>> &P3_out,
                              std::vector<Eigen::Matrix<double, 2, 4>> &P4_out,
                              std::vector<Eigen::Matrix<double, 16, 1>> &QFs) {
    // initialize the variables for the tracking
    std::complex<double> params[228];
    static std::complex<double> solution[8];
    int num_steps;

    int num_good_poses = 0;

    // copy the start problem
    for (int a = 0; a < 114; a++) {
        params[a] = start_system.problem[a];
    }

    // convert the 2D points into the coefficients used by the solver
    Eigen::Matrix<double, 7, 16> coef_matrix;
    Eigen::Matrix2d eps;
    eps << 0, 1, 1, 0;
    for (int a = 0; a < 7; ++a) {
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
            params[114 + 16 * a + b] = c;
            coef_matrix(a, b) = c;
        }
    }
    params[226] = 1;
    params[227] = 1;

    // if the system is badly conditioned, do not compute it
    Eigen::JacobiSVD<Eigen::Matrix<double, 7, 16>> svd(coef_matrix, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix<double, 7, 1> svs = svd.singularValues();
    double cond = svs(6);
    if (cond < 1e-10)
        return 0;

    // track from every solution to the starting system
    for (int k = 0; k < 25; ++k) {
        // copy the solution
        std::complex<double> cur_start[8];
        for (int a = 0; a < 8; ++a) {
            cur_start[a] = start_system.sols[k][a];
        }

        // track the problem
        int status = upright::track(settings, cur_start, params, solution, &num_steps);

        // convert the solutions into cameras and triangulate the 3D points
        if (status == 2) {
            // build the cameras from the IDs
            Eigen::Matrix<double, 2, 4> Ps[4];
            upright_sol2cam(solution, Ps);

            const Eigen::Matrix<double, 16, 1> QF = upright_cams2qft(Ps);

            P1_out.push_back(Ps[0]);
            P2_out.push_back(Ps[1]);
            P3_out.push_back(Ps[2]);
            P4_out.push_back(Ps[3]);
            QFs.push_back(QF);
        }
    }
    return P1_out.size();
}

} // namespace rqt
