#include "nanson2_radial_quadrifocal_solver.h"
#include "nanson2_homotopy.h"

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
//using namespace std;


//decomposes the QF tensor to a set of camera matrices
void nanson2_sol2cam(Eigen::Matrix<double,16,1> QF, Eigen::Matrix<double, 2, 4> *Ps)
{
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


//solves one instance of the radial quadrifocal problem
int nanson2_radial_quadrifocal_solver(const std::vector<Eigen::Vector2d> &p1s, const std::vector<Eigen::Vector2d> &p2s,
                              const std::vector<Eigen::Vector2d> &p3s, const std::vector<Eigen::Vector2d> &p4s,
                              const StartSystem &start_system, const TrackSettings &settings,
                              std::vector<Eigen::Matrix<double, 2, 4>> &P1_out,
                              std::vector<Eigen::Matrix<double, 2, 4>> &P2_out,
                              std::vector<Eigen::Matrix<double, 2, 4>> &P3_out,
                              std::vector<Eigen::Matrix<double, 2, 4>> &P4_out,
                              std::vector<Eigen::Matrix<double, 16, 1>> &QFs)
{
	//initialize the variables for the tracking
	std::complex<double> params[96];
	static std::complex<double> solution[2];
	int num_steps;
	
	int num_good_poses = 0;
	
	//copy the start problem
	for(int i=0;i<48;i++)
	{
		params[i] = start_system.problem[i];
	}

	//convert the 2D points into the coefficients used by the solver
	Eigen::Matrix<double,13,16> coef_matrix;
	Eigen::Matrix2d eps;
	eps << 0,1,1,0;
	for(int i=0;i<13;++i)
	{
		//obtain lines on which the point projections lie
		Eigen::Vector2d l1 = eps*p1s[i];
		l1 = l1/l1.norm();
		Eigen::Vector2d l2 = eps*p2s[i];
		l2 = l2/l2.norm();
		Eigen::Vector2d l3 = eps*p3s[i];
		l3 = l3/l3.norm();
		Eigen::Vector2d l4 = eps*p4s[i];
		l4 = l4/l4.norm();
		
		//generate the linear coefficients for the QF tensor
		for(int j=0;j<16;++j)
		{
			int ix4 = j%2;
			int ix3 = (j >> 1)%2;
			int ix2 = (j >> 2)%2;
			int ix1 = (j >> 3)%2;
			
			double c = l1(ix1)*l2(ix2)*l3(ix3)*l4(ix4);
			coef_matrix(i,j) = c;
		}
	}
	
	//get the kernel of the linear system
	Eigen::JacobiSVD<Eigen::Matrix<double,13,16>> svd(coef_matrix, Eigen::ComputeFullU | Eigen::ComputeFullV);
	Eigen::Matrix<double,13,1> svs = svd.singularValues();
	double cond = svs(12);
	if(cond < 1e-10) return 0;
	
	Eigen::Matrix<double,16,1> QF1 = svd.matrixV().col(13);
	Eigen::Matrix<double,16,1> QF2 = svd.matrixV().col(14);
	Eigen::Matrix<double,16,1> QF3 = svd.matrixV().col(15);
	for(int i=0;i<16;++i)
	{
		params[48+i] = QF1(i);
		params[48+16+i] = QF2(i);
		params[48+32+i] = QF3(i);
	}
	
	//track from every solution to the starting system	
	for(int i=0;i<28;++i)
	{		
		std::complex<double> cur_start[2];
		for(int j=0;j<2;++j)
		{
			cur_start[j] = start_system.sols[i][j];
		}
			
		//track the problem
		int status = nanson2::track(settings, cur_start, params, solution, &num_steps);
		if(status==2)
		{
			//construct the QF tensor
			Eigen::Matrix<double,16,1> QF = solution[0].real()*QF1 + solution[1].real()*QF2 + QF3;
			
			//decompose the QF tensor to get uncalibrated camera matrices
			Eigen::Matrix<double, 2, 4> Ps[32];
			nanson2_sol2cam(QF, Ps);
			
			int num_uncal;
			Eigen::Matrix<double,2,4> P1_uncal[8];
			Eigen::Matrix<double,2,4> P2_uncal[8];
			Eigen::Matrix<double,2,4> P3_uncal[8];
			Eigen::Matrix<double,2,4> P4_uncal[8];
			for(int k=0;k<8;++k)
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

