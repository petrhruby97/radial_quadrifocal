#include "quadrifocal_estimator.h"
#include "radial_quadrifocal_solver.h"
#include "metric_upgrade.h"
#include <iostream>

namespace rqt {

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

    std::cout << "Drew sample: ";
    for(int s : sample)
        std::cout << s << ",";
    std::cout << " out of " << num_data << " points." << std::endl;


    for(int i = 0; i < sample_sz; ++i) {
        x1s[i] = x1[sample[i]];
        x2s[i] = x2[sample[i]];
        x3s[i] = x3[sample[i]];
        x4s[i] = x4[sample[i]];
    }

    // Solve for projective cameras
    std::vector<Eigen::Matrix<double, 2, 4>> P1, P2, P3, P4;
    std::vector<Eigen::Matrix<double, 16, 1>> QFs;
    int num_projective = radial_quadrifocal_solver(x1, x2, x3, x4, start_system, track_settings, P1, P2, P3, P4, QFs);

    // Upgrade to metric
    std::vector<Eigen::Matrix<double, 2, 4>> P1_calib, P2_calib, P3_calib, P4_calib;
    std::vector<std::vector<Eigen::Vector3d>> Xs;
    int total_valid = 0;
    for (int i = 0; i < num_projective; ++i) {
        int valid =
            metric_upgrade(x1, x2, x3, x4, P1[i], P2[i], P3[i], P4[i], P1_calib, P2_calib, P3_calib, P4_calib, Xs);
        total_valid += valid;
    }

    std::cout << "Found " << total_valid << " calibrated factorizations satisfying cheirality." << std::endl;

    models->clear();
    for(int i = 0; i < total_valid; ++i) {        
        std::cout << "Triangulating " << i << "\n";
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
    std::cout << "generate_models done. Found " << models->size() << " models" << std::endl;
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

void QuadrifocalEstimator::refine_model(Reconstruction *rec) const {

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