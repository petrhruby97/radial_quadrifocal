#ifndef RQT_TYPES_H_
#define RQT_TYPES_H_

#include <complex>
#include <string>
#include <vector>
#include <limits>

namespace rqt {

enum class MinimalSolver {
    MINIMAL = 0,
    LINEAR = 1,
    UPRIGHT = 2,
    NANSON = 3,
    NANSON2 = 4
};

struct TrackSettings {
    TrackSettings()
        : init_dt_(0.05),                                          // m2 tStep, t_step, raw interface code initDt
          min_dt_(1e-10),                                          // m2 tStepMin, raw interface code minDt
          end_zone_factor_(0.05), epsilon_(1e-4),                  // m2 CorrectorTolerance
          epsilon2_(epsilon_ * epsilon_), dt_increase_factor_(3.), // m2 stepIncreaseFactor
          dt_decrease_factor_(1. / dt_increase_factor_),           // m2 stepDecreaseFactor not existent in
                                                                   // DEFAULT, using what is in track.m2:77
          infinity_threshold_(1e7),                                // m2 InfinityThreshold
          infinity_threshold2_(infinity_threshold_ * infinity_threshold_),
          max_corr_steps_(3),               // m2 maxCorrSteps (track.m2 param of rawSetParametersPT corresp
                                            // to max_corr_steps in NAG.cpp)
          num_successes_before_increase_(3) // m2 numberSuccessesBeforeIncrease
    {}

    double init_dt_; // m2 tStep, t_step, raw interface code initDt
    double min_dt_;  // m2 tStepMin, raw interface code minDt
    double end_zone_factor_;
    double epsilon_; // m2 CorrectorTolerance (chicago.m2, track.m2), raw interface
                     // code epsilon (interface2.d, NAG.cpp:rawSwetParametersPT)
    double epsilon2_;
    double dt_increase_factor_; // m2 stepIncreaseFactor
    double dt_decrease_factor_; // m2 stepDecreaseFactor not existent in DEFAULT,
                                // using what is in track.m2:77
    double infinity_threshold_; // m2 InfinityThreshold
    double infinity_threshold2_;
    unsigned max_corr_steps_;                // m2 maxCorrSteps (track.m2 param of rawSetParametersPT
                                             // corresp to max_corr_steps in NAG.cpp)
    unsigned num_successes_before_increase_; // m2 numberSuccessesBeforeIncrease
    double corr_thresh_;
    unsigned anch_num_;

    bool load_settings(const std::string &filename);
};

struct StartSystem {
    std::vector<std::complex<double>> problem;
    std::vector<std::vector<std::complex<double>>> sols;

    bool load_start_system(const std::string &filename, MinimalSolver solver);
    bool load_default(MinimalSolver solver);
};


struct RansacOptions {
    size_t max_iterations = 100000;
    size_t min_iterations = 1000;
    double dyn_num_trials_mult = 3.0;
    double success_prob = 0.9999;
    double max_error = 12.0;
    MinimalSolver solver = MinimalSolver::MINIMAL;
};

struct RansacStats {
    size_t refinements = 0;
    size_t iterations = 0;
    size_t num_inliers = 0;
    double inlier_ratio = 0;
    double model_score = std::numeric_limits<double>::max();
};

} // namespace rqt

#endif
