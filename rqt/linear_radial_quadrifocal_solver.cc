#include "linear_radial_quadrifocal_solver.h"

#include "solver_det4.h"

#include <Eigen/Core>
#include <Eigen/Dense>
#include <chrono>
#include <complex>
#include <float.h>
#include <math.h>
#include <string>
#include <vector>

#include <iostream>

namespace rqt {

#define Float double
typedef unsigned char ind;
static constexpr Float tol = 1e-3;
typedef std::complex<Float> complex;
using namespace std::chrono;
// using namespace std;

// FUNCTIONS

// converts the solution to uncalibrated cameras
void sol2cam(Eigen::Matrix<double,16,1> QF, Eigen::Matrix<double, 2, 4> *Ps) {
	double sc = QF(0);

	QF = (QF/sc);
	double p44 = -QF(1);
	double p33 = -QF(2);
	double p22 = -QF(4);
	double p11 = -QF(8);
	double p41 = p44-(QF(9)/p11);
	double p31 = p33-(QF(10)/p11);
	double p21 = p22-(QF(12)/p11);

	double A34 = -p11*p41;
	double B34 = -p11*p31;
	double C34 = p11*p33*p41 + p11*(p33*p44-QF(3)) + p11*p31*p44 - p11*p33*p44 - QF(11);
	double D34 = QF(3)-p33*p44;
	double a34 = -B34/A34;
	double b34 = -C34/A34;
	double c34 = D34;
	double d34 = b34*b34-4*a34*c34;
	double p43A = (-b34+std::sqrt(d34))/(2*a34);
	double p43B = (-b34-std::sqrt(d34))/(2*a34);
	double p34A = (-B34*p43A-C34)/A34;
	double p34B = (-B34*p43B-C34)/A34;

	double A24 = -p11*p41;
	double B24 = -p11*p21;
	double C24 = p11*p22*p41 + p11*(p22*p44-QF(5)) + p11*p21*p44 - p11*p22*p44 - QF(13);
	double D24 = QF(5)-p22*p44;
	double a24 = -B24/A24;
	double b24 = -C24/A24;
	double c24 = D24;
	double d24 = b24*b24-4*a24*c24;
	double p42A = (-b24+std::sqrt(d24))/(2*a24);
	double p42B = (-b24-std::sqrt(d24))/(2*a24);
	double p24A = (-B24*p42A-C24)/A24;
	double p24B = (-B24*p42B-C24)/A24;

	double A23 = -p11*p31;
	double B23 = -p11*p21;
	double C23 = p11*p22*p31 + p11*(p22*p33-QF(6)) + p11*p21*p33 - p11*p22*p33 - QF(14);
	double D23 = QF(6)-p22*p33;
	double a23 = -B23/A23;
	double b23 = -C23/A23;
	double c23 = D23;
	double d23 = b23*b23-4*a23*c23;
	double p32A = (-b23+std::sqrt(d23))/(2*a23);
	double p32B = (-b23-std::sqrt(d23))/(2*a23);
	double p23A = (-B23*p32A-C23)/A23;
	double p23B = (-B23*p32B-C23)/A23;

	//std::cout << p11*p44 - p11*p41 - QF(9) << "\n\n";


	//AAA
    Eigen::Matrix<double, 2, 4> P1 = Eigen::Matrix<double, 2, 4>::Zero();
    P1(0, 0) = 1;
    P1(1, 0) = p11;
    P1(1, 1) = p11;
    P1(1, 2) = p11;
    P1(1, 3) = p11;
    Ps[0] = P1;

    Eigen::Matrix<double, 2, 4> P2XAA = Eigen::Matrix<double, 2, 4>::Zero();
    P2XAA(0, 1) = 1;
    P2XAA(1, 0) = p21;
    P2XAA(1, 1) = p22;
    P2XAA(1, 2) = p23A;
    P2XAA(1, 3) = p24A;
    Ps[1] = P2XAA;

    Eigen::Matrix<double, 2, 4> P3AXA = Eigen::Matrix<double, 2, 4>::Zero();
    P3AXA(0, 2) = 1;
    P3AXA(1, 0) = p31;
    P3AXA(1, 1) = p32A;
    P3AXA(1, 2) = p33;
    P3AXA(1, 3) = p34A;
    Ps[2] = P3AXA;

    Eigen::Matrix<double, 2, 4> P4AAX = Eigen::Matrix<double, 2, 4>::Zero();
    P4AAX(0, 3) = 1;
    P4AAX(1, 0) = p41;
    P4AAX(1, 1) = p42A;
    P4AAX(1, 2) = p43A;
    P4AAX(1, 3) = p44;
    Ps[3] = P4AAX;

    //AAB
    Eigen::Matrix<double, 2, 4> P2XAB = Eigen::Matrix<double, 2, 4>::Zero();
    P2XAB(0, 1) = 1;
    P2XAB(1, 0) = p21;
    P2XAB(1, 1) = p22;
    P2XAB(1, 2) = p23B;
    P2XAB(1, 3) = p24A;

    Eigen::Matrix<double, 2, 4> P3AXB = Eigen::Matrix<double, 2, 4>::Zero();
    P3AXB(0, 2) = 1;
    P3AXB(1, 0) = p31;
    P3AXB(1, 1) = p32B;
    P3AXB(1, 2) = p33;
    P3AXB(1, 3) = p34A;

    Ps[4] = P1;
    Ps[5] = P2XAB;
    Ps[6] = P3AXB;
    Ps[7] = P4AAX;

    //ABA

    Eigen::Matrix<double, 2, 4> P2XBA = Eigen::Matrix<double, 2, 4>::Zero();
    P2XBA(0, 1) = 1;
    P2XBA(1, 0) = p21;
    P2XBA(1, 1) = p22;
    P2XBA(1, 2) = p23A;
    P2XBA(1, 3) = p24B;
    
    Eigen::Matrix<double, 2, 4> P4ABX = Eigen::Matrix<double, 2, 4>::Zero();
    P4ABX(0, 3) = 1;
    P4ABX(1, 0) = p41;
    P4ABX(1, 1) = p42B;
    P4ABX(1, 2) = p43A;
    P4ABX(1, 3) = p44;

    Ps[8] = P1;
    Ps[9] = P2XBA;
    Ps[10] = P3AXA;
    Ps[11] = P4ABX;

    //ABB
   
    Eigen::Matrix<double, 2, 4> P2XBB = Eigen::Matrix<double, 2, 4>::Zero();
    P2XBB(0, 1) = 1;
    P2XBB(1, 0) = p21;
    P2XBB(1, 1) = p22;
    P2XBB(1, 2) = p23B;
    P2XBB(1, 3) = p24B;

    Ps[12] = P1;
    Ps[13] = P2XBB;
    Ps[14] = P3AXB;
    Ps[15] = P4ABX;

    //BAA
    
    Eigen::Matrix<double, 2, 4> P3BXA = Eigen::Matrix<double, 2, 4>::Zero();
    P3BXA(0, 2) = 1;
    P3BXA(1, 0) = p31;
    P3BXA(1, 1) = p32A;
    P3BXA(1, 2) = p33;
    P3BXA(1, 3) = p34B;

    Eigen::Matrix<double, 2, 4> P4BAX = Eigen::Matrix<double, 2, 4>::Zero();
    P4BAX(0, 3) = 1;
    P4BAX(1, 0) = p41;
    P4BAX(1, 1) = p42A;
    P4BAX(1, 2) = p43B;
    P4BAX(1, 3) = p44;

    Ps[16] = P1;
    Ps[17] = P2XAA;
    Ps[18] = P3BXA;
    Ps[19] = P4BAX;

    //BAB
    
    Eigen::Matrix<double, 2, 4> P3BXB = Eigen::Matrix<double, 2, 4>::Zero();
    P3BXB(0, 2) = 1;
    P3BXB(1, 0) = p31;
    P3BXB(1, 1) = p32B;
    P3BXB(1, 2) = p33;
    P3BXB(1, 3) = p34B;

    Ps[20] = P1;
    Ps[21] = P2XAB;
    Ps[22] = P3BXB;
    Ps[23] = P4BAX;

    //BBA
    
    Eigen::Matrix<double, 2, 4> P4BBX = Eigen::Matrix<double, 2, 4>::Zero();
    P4BBX(0, 3) = 1;
    P4BBX(1, 0) = p41;
    P4BBX(1, 1) = p42B;
    P4BBX(1, 2) = p43B;
    P4BBX(1, 3) = p44;

    Ps[24] = P1;
    Ps[25] = P2XBA;
    Ps[26] = P3BXA;
    Ps[27] = P4BBX;

    //BBB
    
    Ps[28] = P1;
    Ps[29] = P2XBB;
    Ps[30] = P3BXB;
    Ps[31] = P4BBX;

}

// creates a symmetric system of cameras from the original system
// DEFINED IN radial_quadrifocal_solver.cc and for some reason visible here
/*void get_symmetric_cams_(Eigen::Matrix<double, 2, 4> *Ps, Eigen::Matrix<double, 2, 4> *Ps_sym) {
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
}*/

// finds a quadrifocal tensor from cameras
/*Eigen::Matrix<double, 16, 1> cams2qft_(Eigen::Matrix<double, 2, 4> *Ps) {
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
}*/

// solves one instance of the radial quadrifocal problem
int linear_radial_quadrifocal_solver(const std::vector<Eigen::Vector2d> &p1s, const std::vector<Eigen::Vector2d> &p2s,
                              const std::vector<Eigen::Vector2d> &p3s, const std::vector<Eigen::Vector2d> &p4s,
                              const StartSystem &start_system, const TrackSettings &settings,
                              std::vector<Eigen::Matrix<double, 2, 4>> &P1_out,
                              std::vector<Eigen::Matrix<double, 2, 4>> &P2_out,
                              std::vector<Eigen::Matrix<double, 2, 4>> &P3_out,
                              std::vector<Eigen::Matrix<double, 2, 4>> &P4_out,
                              std::vector<Eigen::Matrix<double, 16, 1>> &QFs) {
    // initialize the variables for the tracking
    //std::complex<double> params[448];
    int num_good_poses = 0;

    // convert the 2D points into the coefficients used by the solver
    Eigen::Matrix<double, 15, 16> coef_matrix;
    Eigen::Matrix2d eps;
    eps << 0, 1, 1, 0;
    for (int a = 0; a < 15; ++a) {
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
            //params[224 + 16 * a + b] = c;
            coef_matrix(a, b) = c;
        }
    }
    // if the system is badly conditioned, do not compute it
    Eigen::JacobiSVD<Eigen::Matrix<double, 15, 16>> svd(coef_matrix, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix<double, 15, 1> svs = svd.singularValues();
    //std::cout << svs << "\n\n";
    /*double cond = svs(12);
    if (cond < 1e-10)
        return 0;*/
    Eigen::Matrix<double,16,1> QF = svd.matrixV().col(15);
    //std::cout << coef_matrix * QF << "\n\n\n\n"; 
    Eigen::Matrix<double, 2, 4> Ps[32];
    sol2cam(QF, Ps);

    for(int i=0;i<8;++i)
    {
	    //std::cout << QF/QF(0)<< "\n\n";
	    //Eigen::Matrix<double,16,1> QF_ = cams2qft(Ps+4*i);
            //std::cout << QF/QF(0) - QF_/QF_(0) << "\n\n\n\n";
	    P1_out.push_back(Ps[0+4*i]);
	    P2_out.push_back(Ps[1+4*i]);
	    P3_out.push_back(Ps[2+4*i]);
	    P4_out.push_back(Ps[3+4*i]);
	    QFs.push_back(QF);
    }

    // get the symmetric cameras
    /*Eigen::Matrix<double, 2, 4> Ps_sym[4];
    get_symmetric_cams(Ps, Ps_sym);

    P1_out.push_back(Ps_sym[0]);
    P2_out.push_back(Ps_sym[1]);
    P3_out.push_back(Ps_sym[2]);
    P4_out.push_back(Ps_sym[3]);
    QFs.push_back(QF);*/

    // track from every solution to the starting system
    /*for (int k = 0; k < 28; ++k) {
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
    }*/
    return P1_out.size();
}

} // namespace rqt
