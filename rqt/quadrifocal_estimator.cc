#include "quadrifocal_estimator.h"
#include "radial_quadrifocal_solver.h"
#include "linear_radial_quadrifocal_solver.h"
#include "upright_radial_quadrifocal_solver.h"
#include "nanson_radial_quadrifocal_solver.h"
#include "nanson2_radial_quadrifocal_solver.h"
#include "metric_upgrade.h"
#include "upright_filter_cheirality.h"
#include <ceres/ceres.h>

namespace rqt {

// error for the radial projection
struct RadialReprojError { 
    RadialReprojError(const Eigen::Vector2d& x) : x_(x) {}

    template <typename T>
    bool operator()(const T* const qvec, const T* const tvec, const T* const Xvec, T* residuals) const {
        Eigen::Quaternion<T> q;
        q.coeffs() << qvec[0], qvec[1], qvec[2], qvec[3];

        Eigen::Matrix<T, 3, 1> X;
        X << Xvec[0], Xvec[1], Xvec[2];

        Eigen::Matrix<T, 3, 1> Z = q.toRotationMatrix() * X;

        Eigen::Matrix<T, 2, 1> t;
        t << tvec[0], tvec[1];

        Eigen::Matrix<T, 2, 1> z = (Z.template topRows<2>() + t).normalized();

        Eigen::Matrix<T, 2, 1> xc;
        xc << T(x_(0)), T(x_(1));

        T alpha = z.dot(xc);                
        residuals[0] = alpha * z(0) - xc(0);
        residuals[1] = alpha * z(1) - xc(1);
        return true;
    }

    // Factory function
    static ceres::CostFunction* CreateCost(const Eigen::Vector2d &x) {
        return (new ceres::AutoDiffCostFunction<RadialReprojError, 2, 4, 2, 3>(new RadialReprojError(x)));
    }

private:
    const Eigen::Vector2d& x_;
};



void QuadrifocalEstimator::generate_models(std::vector<Reconstruction> *models) {
    std::uniform_int_distribution<int> rand_int(0, num_data);
    sample.clear();
    while(sample.size() < sample_sz) {
        int s = rand_int(rng);
        bool new_sample = true;
        for(int i : sample) {
            if(s == i) {
                new_sample = false;
                break;
            }
        }
        if(new_sample)
            sample.push_back(s);
    }

    for(int i = 0; i < sample_sz; ++i) {
        x1s[i] = x1[sample[i]];
        x2s[i] = x2[sample[i]];
        x3s[i] = x3[sample[i]];
        x4s[i] = x4[sample[i]];
    }

    std::vector<Eigen::Matrix<double, 2, 4>> P1_calib, P2_calib, P3_calib, P4_calib;
    std::vector<std::vector<Eigen::Vector3d>> Xs;

    if(opt.solver == MinimalSolver::MINIMAL) {
        // Solve for projective cameras with minimal solver
        std::vector<Eigen::Matrix<double, 2, 4>> P1, P2, P3, P4;
        std::vector<Eigen::Matrix<double, 16, 1>> QFs;
        int num_projective = radial_quadrifocal_solver(x1s, x2s, x3s, x4s, start_system, track_settings, P1, P2, P3, P4, QFs);

        // Upgrade to metric
        int total_valid = 0;
        for (int i = 0; i < num_projective; ++i) {
            int valid =
                metric_upgrade(x1s, x2s, x3s, x4s, P1[i], P2[i], P3[i], P4[i], P1_calib, P2_calib, P3_calib, P4_calib, Xs);
            total_valid += valid;
        }
    } else if(opt.solver == MinimalSolver::LINEAR) {
	// Solve for projective cameras
        std::vector<Eigen::Matrix<double, 2, 4>> P1, P2, P3, P4;
        std::vector<Eigen::Matrix<double, 16, 1>> QFs;
        int num_projective = linear_radial_quadrifocal_solver(x1s, x2s, x3s, x4s, start_system, track_settings, P1, P2, P3, P4, QFs);

        // Upgrade to metric
        std::vector<std::vector<Eigen::Vector3d>> Xs;
        int total_valid = 0;
        for (int i = 0; i < num_projective; ++i) {
	    int valid =
	        metric_upgrade(x1s, x2s, x3s, x4s, P1[i], P2[i], P3[i], P4[i], P1_calib, P2_calib, P3_calib, P4_calib, Xs);
	    total_valid += valid;
        }
    } else if(opt.solver == MinimalSolver::UPRIGHT) {
        // Solve for projective cameras
        std::vector<Eigen::Matrix<double, 2, 4>> P1, P2, P3, P4;
        std::vector<Eigen::Matrix<double, 16, 1>> QFs;
        int num_projective = upright_radial_quadrifocal_solver(x1s, x2s, x3s, x4s, start_system, track_settings, P1, P2, P3, P4, QFs);
	
	// Upgrade to metric
	std::vector<std::vector<Eigen::Vector3d>> Xs;
	int total_valid = 0;
	for (int i = 0; i < num_projective; ++i) {
	    int valid =
		upright_filter_cheirality(x1s, x2s, x3s, x4s, P1[i], P2[i], P3[i], P4[i], P1_calib, P2_calib, P3_calib, P4_calib, Xs);
	    total_valid += valid;
	}
    } else if(opt.solver == MinimalSolver::NANSON) {
        // Solve for projective cameras with minimal solver
        std::vector<Eigen::Matrix<double, 2, 4>> P1, P2, P3, P4;
        std::vector<Eigen::Matrix<double, 16, 1>> QFs;
        int num_projective = nanson_radial_quadrifocal_solver(x1s, x2s, x3s, x4s, start_system, track_settings, P1, P2, P3, P4, QFs);

        // Upgrade to metric
        int total_valid = 0;
        for (int i = 0; i < num_projective; ++i) {
            int valid =
                metric_upgrade(x1s, x2s, x3s, x4s, P1[i], P2[i], P3[i], P4[i], P1_calib, P2_calib, P3_calib, P4_calib, Xs);
            total_valid += valid;
        }
    } else if(opt.solver == MinimalSolver::NANSON2) {
        // Solve for projective cameras with minimal solver
        std::vector<Eigen::Matrix<double, 2, 4>> P1, P2, P3, P4;
        std::vector<Eigen::Matrix<double, 16, 1>> QFs;
        int num_projective = nanson2_radial_quadrifocal_solver(x1s, x2s, x3s, x4s, start_system, track_settings, P1, P2, P3, P4, QFs);

        // Upgrade to metric
        int total_valid = 0;
        for (int i = 0; i < num_projective; ++i) {
            int valid =
                metric_upgrade(x1s, x2s, x3s, x4s, P1[i], P2[i], P3[i], P4[i], P1_calib, P2_calib, P3_calib, P4_calib, Xs);
            total_valid += valid;
        }
    }

    models->clear();
    for(int i = 0; i < P1_calib.size(); ++i) {        
        Reconstruction rec;
        rec.P1 = P1_calib[i];
        rec.P2 = P2_calib[i];
        rec.P3 = P3_calib[i];
        rec.P4 = P4_calib[i];

        // Ensure we actually got rotations back from the solver
        const double max_rot_error = 1e-6;
        const double e1 = (rec.P1.block<2,3>(0,0) * rec.P1.block<2,3>(0,0).transpose() - Eigen::Matrix2d::Identity()).norm();
        const double e2 = (rec.P2.block<2,3>(0,0) * rec.P2.block<2,3>(0,0).transpose() - Eigen::Matrix2d::Identity()).norm();
        const double e3 = (rec.P3.block<2,3>(0,0) * rec.P3.block<2,3>(0,0).transpose() - Eigen::Matrix2d::Identity()).norm();
        const double e4 = (rec.P4.block<2,3>(0,0) * rec.P4.block<2,3>(0,0).transpose() - Eigen::Matrix2d::Identity()).norm();
        if(e1 > max_rot_error || e2 > max_rot_error || e3 > max_rot_error || e4 > max_rot_error) {
            continue;
        }

        triangulate(rec);
        models->push_back(rec);
    }
    //std::cout << P1_calib.size() << " " << models->size() << "\n";
}

double QuadrifocalEstimator::score_model(Reconstruction &rec, size_t *inlier_count) const {
    const double sqr_thr = opt.max_error * opt.max_error;

    double score = 0;
    *inlier_count = 0;
    for(int i = 0; i < num_data; ++i) {
        Eigen::Vector4d Xh = rec.X[i].homogeneous();
        Eigen::Vector2d z1 = (rec.P1 * Xh).normalized();
        Eigen::Vector2d z2 = (rec.P2 * Xh).normalized();
        Eigen::Vector2d z3 = (rec.P3 * Xh).normalized();
        Eigen::Vector2d z4 = (rec.P4 * Xh).normalized();

        const double alpha1 = z1.dot(x1[i]);
        const double alpha2 = z2.dot(x2[i]);
        const double alpha3 = z3.dot(x3[i]);
        const double alpha4 = z4.dot(x4[i]);

        const bool bad_cheiral = alpha1 < 0 || alpha2 < 0 || alpha3 < 0 || alpha4 < 0;

        double r1 = (x1[i] - alpha1 * z1).squaredNorm();
        double r2 = (x2[i] - alpha2 * z2).squaredNorm();
        double r3 = (x3[i] - alpha3 * z3).squaredNorm();
        double r4 = (x4[i] - alpha4 * z4).squaredNorm();

        const bool inlier = r1 < sqr_thr && r2 < sqr_thr && r3 < sqr_thr && r4 < sqr_thr;

        if(inlier && !bad_cheiral) {
            score += r1 + r2 + r3 + r4;
            rec.inlier[i] = true;
            (*inlier_count)++;
        } else {
            rec.inlier[i] = false;
            score += 4 * sqr_thr;
        }
    }
    return score;
}


Eigen::Matrix3d complete_rotation(const Eigen::Matrix<double,2,3> &R_2x3) {
    Eigen::Matrix3d R;
    R.block<2,3>(0,0) = R_2x3;
    R.row(0).normalize();
    // We orthogonalize here just in case
    R.row(1) = R.row(1) - R.row(1).dot(R.row(0)) * R.row(0);
    R.row(1).normalize();
    R.row(2) = R.row(0).cross(R.row(1));    
    return R;
}

void QuadrifocalEstimator::refine_model(Reconstruction *rec) const {
    Eigen::Quaterniond q1(complete_rotation(rec->P1.block<2,3>(0,0)));
    Eigen::Quaterniond q2(complete_rotation(rec->P2.block<2,3>(0,0)));
    Eigen::Quaterniond q3(complete_rotation(rec->P3.block<2,3>(0,0)));
    Eigen::Quaterniond q4(complete_rotation(rec->P4.block<2,3>(0,0)));

    Eigen::Vector2d t1 = rec->P1.col(3);
    Eigen::Vector2d t2 = rec->P2.col(3);
    Eigen::Vector2d t3 = rec->P3.col(3);
    Eigen::Vector2d t4 = rec->P4.col(3);

    
    ceres::Problem problem;
    //ceres::LossFunction* loss_function = nullptr;;
    ceres::LossFunction* loss_function = new ceres::TrivialLoss();
    ceres::CostFunction* cost;

    //OLD VERSION USES THIS INSTEAD
    /*ceres::Problem problem;
    ceres::LossFunction* loss_function = nullptr;;
    ceres::CostFunction* cost;*/

    int num_inliers = 0;
    for(int i = 0; i < num_data; ++i) {
        if(!rec->inlier[i]) {
            continue;
        }

        problem.AddResidualBlock(RadialReprojError::CreateCost(x1[i]), loss_function, q1.coeffs().data(), t1.data(), rec->X[i].data());
        problem.AddResidualBlock(RadialReprojError::CreateCost(x2[i]), loss_function, q2.coeffs().data(), t2.data(), rec->X[i].data());
        problem.AddResidualBlock(RadialReprojError::CreateCost(x3[i]), loss_function, q3.coeffs().data(), t3.data(), rec->X[i].data());
        problem.AddResidualBlock(RadialReprojError::CreateCost(x4[i]), loss_function, q4.coeffs().data(), t4.data(), rec->X[i].data());

        num_inliers += 1;
    }
    
    // Minimum number of inliers for the refinement to be well-posed
    // THIS IS NOT COMPLETELY CORRECT BUT IT IS USED IN THE OLD VERSION -> NOT CAUSE OF THE PROBLEM
    if(num_inliers <= 13) {
        return;
    }
      
    problem.SetParameterization(q1.coeffs().data(), new ceres::EigenQuaternionParameterization());
    problem.SetParameterization(q2.coeffs().data(), new ceres::EigenQuaternionParameterization());
    problem.SetParameterization(q3.coeffs().data(), new ceres::EigenQuaternionParameterization());
    problem.SetParameterization(q4.coeffs().data(), new ceres::EigenQuaternionParameterization());

    // Fix gauge-freedom
    problem.SetParameterBlockConstant(q1.coeffs().data());
    problem.SetParameterBlockConstant(t1.data());
    problem.SetParameterization(t2.data(), new ceres::SubsetParameterization(2, {0}));

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_SCHUR;
    options.minimizer_progress_to_stdout = false;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    // Update reconstruction with new cameras
    rec->P1.block<2,3>(0,0) = q1.toRotationMatrix().block<2,3>(0,0);
    rec->P1.col(3) = t1;
    rec->P2.block<2,3>(0,0) = q2.toRotationMatrix().block<2,3>(0,0);
    rec->P2.col(3) = t2;
    rec->P3.block<2,3>(0,0) = q3.toRotationMatrix().block<2,3>(0,0);
    rec->P3.col(3) = t3;
    rec->P4.block<2,3>(0,0) = q4.toRotationMatrix().block<2,3>(0,0);
    rec->P4.col(3) = t4;
}


void QuadrifocalEstimator::triangulate(Reconstruction &rec) {
    rec.X.resize(num_data);
    rec.inlier.resize(num_data);
    Eigen::Matrix2d eps2;
    eps2 << 0, 1, -1, 0;        
    for (int k = 0; k < num_data; ++k) {
        // obtain the system whose solution is the homogeneous 3D point
        Eigen::Matrix4d A;
        A.row(0) = x1[k].transpose() * eps2 * rec.P1;
        A.row(1) = x2[k].transpose() * eps2 * rec.P2;
        A.row(2) = x3[k].transpose() * eps2 * rec.P3;
        A.row(3) = x4[k].transpose() * eps2 * rec.P4;

        // use SVD to obtain the kernel of the matrix
        Eigen::JacobiSVD<Eigen::Matrix4d> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Matrix4d V = svd.matrixV();
        rec.X[k] = V.col(3).hnormalized();
    }
}


}
