#ifndef RQT_QUADRIFOCAL_ESTIMATOR_H_
#define RQT_QUADRIFOCAL_ESTIMATOR_H_

#include <Eigen/Dense>
#include <random>
#include "types.h"

namespace rqt {


class QuadrifocalEstimator {
  public:
    QuadrifocalEstimator(const RansacOptions &ransac_opt,
                         const std::vector<Eigen::Vector2d> &points2D_1,
                         const std::vector<Eigen::Vector2d> &points2D_2,
                         const std::vector<Eigen::Vector2d> &points2D_3,
                         const std::vector<Eigen::Vector2d> &points2D_4,
                         const StartSystem &ss, const TrackSettings &ts)
        : sample_sz(get_sample_sz(ransac_opt.solver)), num_data(points2D_1.size()), opt(ransac_opt),
          x1(points2D_1), x2(points2D_2), x3(points2D_3), x4(points2D_4),
          start_system(ss), track_settings(ts) {
        x1s.resize(sample_sz);
        x2s.resize(sample_sz);
        x3s.resize(sample_sz);
        x4s.resize(sample_sz);
        sample.resize(sample_sz);
    }

    struct Reconstruction {
        Eigen::Matrix<double,2,4> P1, P2, P3, P4;
        std::vector<Eigen::Vector3d> X;
        std::vector<bool> inlier;
    };
    typedef Reconstruction model_t;

    void generate_models(std::vector<Reconstruction> *models);
    double score_model(Reconstruction &rec, size_t *inlier_count) const;
    void refine_model(Reconstruction *rec) const;
    void triangulate(Reconstruction &rec);

    const size_t sample_sz;
    const size_t num_data;

  private:
    size_t get_sample_sz(MinimalSolver solv) const {
      if(solv == MinimalSolver::MINIMAL) {
        return 13;
      } else if(solv == MinimalSolver::LINEAR) {
        return 15;
      } else if(solv == MinimalSolver::UPRIGHT) {
        return 7;
      } else if(solv == MinimalSolver::NANSON) {
        return 13;
      } else if(solv == MinimalSolver::NANSON2) {
        return 13;
      } else {
        return -1;
      }
    }

  private:
    const RansacOptions &opt;
    const std::vector<Eigen::Vector2d> &x1;
    const std::vector<Eigen::Vector2d> &x2;
    const std::vector<Eigen::Vector2d> &x3;
    const std::vector<Eigen::Vector2d> &x4;
    const StartSystem &start_system;
    const TrackSettings &track_settings;

    std::mt19937 rng;
    // pre-allocated vectors for sampling
    std::vector<Eigen::Vector2d> x1s, x2s, x3s, x4s;
    std::vector<size_t> sample;
};

}

#endif
