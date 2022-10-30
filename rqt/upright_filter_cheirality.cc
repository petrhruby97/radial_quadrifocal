#include "upright_filter_cheirality.h"

namespace rqt {

int upright_filter_cheirality(const std::vector<Eigen::Vector2d> &x1, const std::vector<Eigen::Vector2d> &x2,
                      const std::vector<Eigen::Vector2d> &x3, const std::vector<Eigen::Vector2d> &x4,
                      const Eigen::Matrix<double, 2, 4> &P1, const Eigen::Matrix<double, 2, 4> &P2,
                      const Eigen::Matrix<double, 2, 4> &P3, const Eigen::Matrix<double, 2, 4> &P4,
                      std::vector<Eigen::Matrix<double, 2, 4>> &P1_calib,
                      std::vector<Eigen::Matrix<double, 2, 4>> &P2_calib,
                      std::vector<Eigen::Matrix<double, 2, 4>> &P3_calib,
                      std::vector<Eigen::Matrix<double, 2, 4>> &P4_calib,
                      std::vector<std::vector<Eigen::Vector3d>> &Xs)
{
	Eigen::Matrix2d eps2;
	eps2 << 0, 1, -1, 0;
	std::vector<Eigen::Vector3d> X(7);
	std::vector<Eigen::Vector3d> Xf(7);
	double s1, s2, s3, s4;
	for (int k = 0; k < 7; ++k)
	{
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

		Eigen::Vector3d cur_Xf = X[k];
		cur_Xf(2) = -1*cur_Xf(2);
		Xf[k] = cur_Xf;

		Eigen::Vector2d pr1 = P1 * X[k].homogeneous();
		Eigen::Vector2d pr2 = P2 * X[k].homogeneous();
		Eigen::Vector2d pr3 = P3 * X[k].homogeneous();
		Eigen::Vector2d pr4 = P4 * X[k].homogeneous();

		const double alpha1 = pr1.dot(x1[k]);
		const double alpha2 = pr2.dot(x2[k]);
		const double alpha3 = pr3.dot(x3[k]);
		const double alpha4 = pr4.dot(x4[k]);

		if (k == 0)
		{
			s1 = alpha1 > 0 ? 1 : -1;
			s2 = alpha2 > 0 ? 1 : -1;
			s3 = alpha3 > 0 ? 1 : -1;
			s4 = alpha4 > 0 ? 1 : -1;
		}
		else
		{
			if (s1 * alpha1 < 0 || s2 * alpha2 < 0 || s3 * alpha3 < 0 || s4 * alpha4 < 0)
			{
				return 0;
			}
		}
	}

	Eigen::Matrix4d H;
	H.setIdentity();

	Eigen::Matrix4d H2;
	H2.setIdentity();
	H2(2,2) = -1;

	if (s1 < 0)
	{
		// ensures that P1 = [I_2, 0]
		H(0, 0) = H(1, 1) = -1;
		for (Eigen::Vector3d &Xk : X)
		{
			Xk.topRows<2>() *= -1;
		}

		H2(0, 0) = H2(1, 1) = -1;
		for (Eigen::Vector3d &Xk : Xf)
		{
			Xk.topRows<2>() *= -1;
		}
	}

	//push back the original solution
	P1_calib.push_back(P1);
	P2_calib.push_back(s2 * P2 * H);
	P3_calib.push_back(s3 * P3 * H);
	P4_calib.push_back(s4 * P4 * H);
	Xs.push_back(X);

	//push back the solution with flipped third column
	P1_calib.push_back(P1);
	P2_calib.push_back(s2 * P2 * H2);
	P3_calib.push_back(s3 * P3 * H2);
	P4_calib.push_back(s4 * P4 * H2);
	Xs.push_back(Xf);


	return 2;
}

} // namespace rqt
