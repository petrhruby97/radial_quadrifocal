// \author Petr Hruby and Viktor Larsson
#include "upright_nanson_radial_quadrifocal_solver.h"

#include "upright_nanson_homotopy.h"

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
void upright_nanson_sol2cam(Eigen::Matrix<double,16,1> QF, Eigen::Matrix<double,2,4> *Ps)
{
	double sc = QF(4);
        QF = QF/sc;

        double p44 = -QF(5);
        double p33 = -QF(6);
        double p22 = -QF(0);
        double p11 = -QF(12);
        double p41 = p44-(QF(13)/p11);
        double p31 = p33-(QF(14)/p11);
        double p21 = p22-(QF(8)/p11);

        double A34 = -p11*p41;
        double B34 = -p11*p31;
        double C34 = p11*p33*p41 + p11*(p33*p44-QF(7)) + p11*p31*p44 - p11*p33*p44 - QF(15);
        double D34 = QF(7)-p33*p44;
        double a34 = -B34/A34;
        double b34 = -C34/A34;
        double c34 = D34;
        double p43 = (-b34)/(2*a34);
        double p34 = (-B34*p43-C34)/A34;

        double A24 = -p11*p41;
        double B24 = -p11*p21;
        double C24 = p11*p22*p41 + p11*(p22*p44-QF(1)) + p11*p21*p44 - p11*p22*p44 - QF(9);
        double D24 = QF(1)-p22*p44;
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
        double C23 = p11*p22*p31 + p11*(p22*p33-QF(2)) + p11*p21*p33 - p11*p22*p33 - QF(10);
        double D23 = QF(2)-p22*p33;
        double a23 = -B23/A23;
        double b23 = -C23/A23;
        double c23 = D23;
        double d23 = b23*b23-4*a23*c23;
        double p32A = (-b23+std::sqrt(d23))/(2*a23);
        double p32B = (-b23-std::sqrt(d23))/(2*a23);
        double p23A = (-B23*p32A-C23)/A23;
        double p23B = (-B23*p32B-C23)/A23;

        Eigen::Matrix<double, 2, 4> P1 = Eigen::Matrix<double, 2, 4>::Zero();
        P1(0, 0) = 1;
        P1(1, 0) = p11;
        P1(1, 1) = p11;
        P1(1, 2) = p11;
        P1(1, 3) = p11;

        Eigen::Matrix<double, 2, 4> P2AA = Eigen::Matrix<double, 2, 4>::Zero();
        P2AA(0, 0) = p21;
        P2AA(0, 1) = p22;
        P2AA(0, 2) = p23A;
        P2AA(0, 3) = p24A;
        P2AA(1, 1) = 1;

        Eigen::Matrix<double, 2, 4> P2AB = Eigen::Matrix<double, 2, 4>::Zero();
        P2AB(0, 0) = p21;
        P2AB(0, 1) = p22;
        P2AB(0, 2) = p23A;
        P2AB(0, 3) = p24B;
        P2AB(1, 1) = 1;

        Eigen::Matrix<double, 2, 4> P2BA = Eigen::Matrix<double, 2, 4>::Zero();
        P2BA(0, 0) = p21;
        P2BA(0, 1) = p22;
        P2BA(0, 2) = p23B;
        P2BA(0, 3) = p24A;
        P2BA(1, 1) = 1;

        Eigen::Matrix<double, 2, 4> P2BB = Eigen::Matrix<double, 2, 4>::Zero();
        P2BB(0, 0) = p21;
        P2BB(0, 1) = p22;
        P2BB(0, 2) = p23B;
        P2BB(0, 3) = p24B;
        P2BB(1, 1) = 1;

	Eigen::Matrix<double, 2, 4> P3AX = Eigen::Matrix<double, 2, 4>::Zero();
        P3AX(0, 2) = 1;
        P3AX(1, 0) = p31;
        P3AX(1, 1) = p32A;
        P3AX(1, 2) = p33;
        P3AX(1, 3) = p34;

        Eigen::Matrix<double, 2, 4> P3BX = Eigen::Matrix<double, 2, 4>::Zero();
        P3BX(0, 2) = 1;
        P3BX(1, 0) = p31;
        P3BX(1, 1) = p32B;
        P3BX(1, 2) = p33;
        P3BX(1, 3) = p34;

        Eigen::Matrix<double, 2, 4> P4XA = Eigen::Matrix<double, 2, 4>::Zero();
        P4XA(0, 3) = 1;
        P4XA(1, 0) = p41;
        P4XA(1, 1) = p42A;
        P4XA(1, 2) = p43;
        P4XA(1, 3) = p44;

        Eigen::Matrix<double, 2, 4> P4XB = Eigen::Matrix<double, 2, 4>::Zero();
        P4XB(0, 3) = 1;
        P4XB(1, 0) = p41;
        P4XB(1, 1) = p42B;
        P4XB(1, 2) = p43;
        P4XB(1, 3) = p44;

        //AA
        Ps[0] = P1;
        Ps[1] = P2AA;
        Ps[2] = P3AX;
        Ps[3] = P4XA;

        //AB
        Ps[4] = P1;
        Ps[5] = P2AB;
        Ps[6] = P3AX;
        Ps[7] = P4XB;

        //BA
        Ps[8] = P1;
        Ps[9] = P2BA;
        Ps[10] = P3BX;
        Ps[11] = P4XA;

        //BB
        Ps[12] = P1;
        Ps[13] = P2BB;
        Ps[14] = P3BX;
        Ps[15] = P4XB;

}

// finds a quadrifocal tensor from cameras
Eigen::Matrix<double, 16, 1> upright_nanson_cams2qft(Eigen::Matrix<double, 2, 4> *Ps) {
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
int upright_nanson_radial_quadrifocal_solver(const std::vector<Eigen::Vector2d> &p1s, const std::vector<Eigen::Vector2d> &p2s,
                              const std::vector<Eigen::Vector2d> &p3s, const std::vector<Eigen::Vector2d> &p4s,
                              const StartSystem &start_system, const TrackSettings &settings,
                              std::vector<Eigen::Matrix<double, 2, 4>> &P1_out,
                              std::vector<Eigen::Matrix<double, 2, 4>> &P2_out,
                              std::vector<Eigen::Matrix<double, 2, 4>> &P3_out,
                              std::vector<Eigen::Matrix<double, 2, 4>> &P4_out,
                              std::vector<Eigen::Matrix<double, 16, 1>> &QFs) {
    // initialize the variables for the tracking
	std::complex<double> params[96];
        static std::complex<double> solution[2];
        int num_steps;


    int num_good_poses = 0;

    // copy the start problem
    for (int a = 0; a < 48; a++) {
        params[a] = start_system.problem[a];
    }

    int ixs[10] = {1,2,3,4,5,6,8,9,10,12};

    // convert the 2D points into the coefficients used by the solver
    Eigen::Matrix<double, 7, 10> coef_matrix;
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

	//generate the coefficients of the problem
        for(int b=0;b<10;++b)
        {
                int ix = ixs[b];
                int ix4 = ix%2;
                int ix3 = (ix >> 1)%2;
                int ix2 = (ix >> 2)%2;
                int ix1 = (ix >> 3)%2;

                double c = l1(ix1)*l2(ix2)*l3(ix3)*l4(ix4);
                coef_matrix(a,b) = c;
        }
    }

    //find the kernel of the system
	Eigen::JacobiSVD<Eigen::Matrix<double,7,10>> svd(coef_matrix, Eigen::ComputeFullU | Eigen::ComputeFullV);
	Eigen::Matrix<double,7,1> svs = svd.singularValues();
	double cond = svs(6);
	if(cond < 1e-10) return 0;

	Eigen::Matrix<double,10,1> QF1r = svd.matrixV().col(7);
	Eigen::Matrix<double,10,1> QF2r = svd.matrixV().col(8);
	Eigen::Matrix<double,10,1> QF3r = svd.matrixV().col(9);
	for(int i=0;i<48;++i)
		params[48+i] = 0;
	for(int i=0;i<10;++i)
	{
		params[48+ixs[i]] = QF1r(i);
		params[48+16+ixs[i]] = QF2r(i);
		params[48+32+ixs[i]] = QF3r(i);
	}

	//find the full representation of the kernel
        Eigen::Matrix<double,16,1> QF1 = Eigen::Matrix<double,16,1>::Zero();
        Eigen::Matrix<double,16,1> QF2 = Eigen::Matrix<double,16,1>::Zero();
        Eigen::Matrix<double,16,1> QF3 = Eigen::Matrix<double,16,1>::Zero();
        for(int i=0;i<10;++i)
        {
                QF1(ixs[i]) = QF1r(i);
                QF2(ixs[i]) = QF2r(i);
                QF3(ixs[i]) = QF3r(i);
        }

	//find the solutions
        for(int i=0;i<25;++i)
        {
                //copy the solution
                std::complex<double> cur_start[2];
                for(int a=0;a<2;++a)
                {
                        cur_start[a] = start_system.sols[i][a];
                }

                //track the problem
                int status = upright_nanson::track(settings, cur_start, params, solution, &num_steps);
                //convert the solutions into cameras and triangulate the 3D points
                if(status == 2)
                {
                        Eigen::Matrix<double,16,1> QF = solution[0].real()*QF1 + solution[1].real()*QF2 + QF3;

                        //decompose the QF tensor to get uncalibrated camera matrices
                        Eigen::Matrix<double, 2, 4> Ps[16];
                        upright_nanson_sol2cam(QF, Ps);

                        for(int k=0;k<4;++k)
                        {
                                P1_out.push_back(Ps[0+4*i]);
                                P2_out.push_back(Ps[1+4*i]);
                                P3_out.push_back(Ps[2+4*i]);
                                P4_out.push_back(Ps[3+4*i]);
                                QFs.push_back(QF);
                        }
                }
        }

    return P1_out.size();
}

} // namespace rqt
