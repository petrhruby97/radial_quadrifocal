#include "rqt/metric_upgrade.h"
#include "rqt/upright_filter_cheirality.h"
#include "rqt/radial_quadrifocal_solver.h"
#include "rqt/linear_radial_quadrifocal_solver.h"
#include "rqt/upright_radial_quadrifocal_solver.h"
#include "rqt/nanson_radial_quadrifocal_solver.h"
#include "rqt/nanson2_radial_quadrifocal_solver.h"
#include "rqt/types.h"
#include "rqt/quadrifocal_estimator.h"
#include "rqt/ransac_impl.h"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <map>
#include <pybind11/eigen.h>
#include <pybind11/iostream.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>


namespace py = pybind11;
using namespace rqt;

template <typename T> void update(const py::dict &input, const std::string &name, T &value) {
    if (input.contains(name)) {
        value = input[name.c_str()].cast<T>();
    }
}
template <> void update(const py::dict &input, const std::string &name, bool &value) {
    if (input.contains(name)) {
        py::object input_value = input[name.c_str()];
        value = (py::str(input_value).is(py::str(Py_True)));
    }
}

TrackSettings settings_from_dict(const py::dict &dict) {
    TrackSettings settings;
    update(dict, "init_dt_", settings.init_dt_);
    update(dict, "min_dt_", settings.min_dt_);
    update(dict, "end_zone_factor_", settings.end_zone_factor_);
    update(dict, "epsilon_", settings.epsilon_);
    update(dict, "epsilon2_", settings.epsilon2_);
    update(dict, "dt_increase_factor_", settings.dt_increase_factor_);
    update(dict, "dt_decrease_factor_", settings.dt_decrease_factor_);
    update(dict, "infinity_threshold_", settings.infinity_threshold_);
    update(dict, "infinity_threshold2_", settings.infinity_threshold2_);
    update(dict, "max_corr_steps_", settings.max_corr_steps_);
    update(dict, "num_successes_before_increase_", settings.num_successes_before_increase_);
    update(dict, "corr_thresh_", settings.corr_thresh_);
    update(dict, "anch_num_", settings.anch_num_);

    return settings;
}

py::dict dict_from_ransac_stats(const RansacStats &stats) {
    py::dict result;
    result["refinements"] = stats.refinements;
    result["iterations"] = stats.iterations;
    result["num_inliers"] = stats.num_inliers;
    result["inlier_ratio"] = stats.inlier_ratio;
    result["model_score"] = stats.model_score;

    return result;
}

MinimalSolver solver_from_string(const std::string &solv) {
    std::string solv_str = solv;
    for (char &c : solv_str)
        c = std::toupper(c);
    if(solv_str == "MINIMAL") {
        return MinimalSolver::MINIMAL;
    } else if(solv_str == "LINEAR") {
        return MinimalSolver::LINEAR;
    } else if(solv_str == "UPRIGHT") {
        return MinimalSolver::UPRIGHT;
    } else if(solv_str == "NANSON") {
        return MinimalSolver::NANSON;
    } else if(solv_str == "NANSON2") {
        return MinimalSolver::NANSON2;
    }
    return MinimalSolver::MINIMAL; // default
}

RansacOptions ransac_options_from_dict(const py::dict &opt_dict) {
    RansacOptions ransac_options;
    update(opt_dict, "max_iterations", ransac_options.max_iterations);
    update(opt_dict, "min_iterations", ransac_options.min_iterations);
    update(opt_dict, "dyn_num_trials_mult", ransac_options.dyn_num_trials_mult);
    update(opt_dict, "success_prob", ransac_options.success_prob);
    update(opt_dict, "max_error", ransac_options.max_error);

    if (opt_dict.contains("solver")) {
        std::string solv_str = opt_dict["solver"].cast<std::string>();
        ransac_options.solver = solver_from_string(solv_str);
    }

    //update(opt_dict, "seed", ransac_options.seed);
    return ransac_options;
}

/*


Camera camera_from_dict(const py::dict &camera_dict) {
    Camera camera;
    camera.model_id =
Camera::id_from_string(camera_dict["model"].cast<std::string>());

    update(camera_dict, "width", camera.width);
    update(camera_dict, "height", camera.height);

    camera.params = camera_dict["params"].cast<std::vector<double>>();
    return camera;
}


static std::string to_string(const Eigen::MatrixXd& mat){
    std::stringstream ss;
    ss << mat;
    return ss.str();
}

template<typename T>
static std::string to_string(const std::vector<T>& vec){
    std::stringstream ss;
    ss << "[";
    if (vec.size() > 0) {
        for (size_t k = 0; k < vec.size() - 1; ++k) {
            ss << vec.at(k) << ", ";
        }
        ss << vec.at(vec.size()-1);
    }
    ss << "]";
    return ss.str();
}

template<typename T>
static std::string vec_to_string(const std::vector<std::vector<T>>& vec){
    std::stringstream ss;
    ss << "[";
    if (vec.size() > 0) {
        for (size_t k = 0; k < vec.size() - 1; ++k) {
            ss << to_string(vec.at(k)) << ", ";
        }
        ss << to_string(vec.at(vec.size()-1));
    }
    ss << "]";
    return ss.str();
}
*/

py::dict ransac_quadrifocal_wrapper(const std::vector<Eigen::Vector2d> &x1,
                                           const std::vector<Eigen::Vector2d> &x2,
                                           const std::vector<Eigen::Vector2d> &x3,
                                           const std::vector<Eigen::Vector2d> &x4, const py::dict &opt) {
    
    RansacOptions ransac_opt = ransac_options_from_dict(opt);
    TrackSettings track_settings;
    StartSystem start_system;

    if (opt.contains("start_system_file")) {
        std::string filename = opt["start_system_file"].cast<std::string>();
        start_system.load_start_system(filename,ransac_opt.solver);
    } else {
        start_system.load_default(ransac_opt.solver);
    }

    if (opt.contains("track_settings_file")) {
        std::string filename = opt["track_settings_file"].cast<std::string>();

        track_settings.load_settings(filename);
    } else {
        track_settings = settings_from_dict(opt);
    }

    QuadrifocalEstimator estimator(ransac_opt,x1,x2,x3,x4,start_system,track_settings);
    QuadrifocalEstimator::Reconstruction best_model;

    auto start_time = std::chrono::high_resolution_clock::now();
    RansacStats stats = ransac(estimator, ransac_opt, &best_model);
    auto end_time = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> runtime_ms = end_time - start_time;

    py::dict result;
    result["P1"] = best_model.P1;
    result["P2"] = best_model.P2;
    result["P3"] = best_model.P3;
    result["P4"] = best_model.P4;
    result["X"] = best_model.X;
    result["inliers"] = best_model.inlier;
    result["runtime"] = runtime_ms.count();
    result["ransac"] = dict_from_ransac_stats(stats);

    return result;
}

py::dict calibrated_radial_quadrifocal_solver_wrapper(const std::vector<Eigen::Vector2d> &x1,
                                                      const std::vector<Eigen::Vector2d> &x2,
                                                      const std::vector<Eigen::Vector2d> &x3,
                                                      const std::vector<Eigen::Vector2d> &x4, const py::dict &opt) {
    MinimalSolver solver = MinimalSolver::MINIMAL;
    if(opt.contains("solver")) {
        solver = solver_from_string(opt["solver"].cast<std::string>());
    }

    TrackSettings track_settings;
    StartSystem start_system;

    if (opt.contains("start_system_file")) {
        std::string filename = opt["start_system_file"].cast<std::string>();
        start_system.load_start_system(filename,solver);
    } else {
        start_system.load_default(solver);
    }

    if (opt.contains("track_settings_file")) {
        std::string filename = opt["track_settings_file"].cast<std::string>();

        track_settings.load_settings(filename);
    } else {
        track_settings = settings_from_dict(opt);
    }

    std::vector<Eigen::Matrix<double, 2, 4>> P1_calib, P2_calib, P3_calib, P4_calib;
    std::vector<Eigen::Matrix<double, 16, 1>> QFs;
    std::vector<std::vector<Eigen::Vector3d>> Xs;

    auto start_time = std::chrono::high_resolution_clock::now();
    int total_valid = 0;

    if(solver == MinimalSolver::MINIMAL) {
        std::vector<Eigen::Matrix<double, 2, 4>> P1, P2, P3, P4;
        int num_projective = radial_quadrifocal_solver(x1, x2, x3, x4, start_system, track_settings, P1, P2, P3, P4, QFs);

        for (int i = 0; i < num_projective; ++i) {
            int valid =
                metric_upgrade(x1, x2, x3, x4, P1[i], P2[i], P3[i], P4[i], P1_calib, P2_calib, P3_calib, P4_calib, Xs);
            total_valid += valid;
        }
    } else if(solver == MinimalSolver::LINEAR) {
        std::vector<Eigen::Matrix<double, 2, 4>> P1, P2, P3, P4;
        int num_projective = linear_radial_quadrifocal_solver(x1, x2, x3, x4, start_system, track_settings, P1, P2, P3, P4, QFs);

        for (int i = 0; i < num_projective; ++i) {
            int valid =
                metric_upgrade(x1, x2, x3, x4, P1[i], P2[i], P3[i], P4[i], P1_calib, P2_calib, P3_calib, P4_calib, Xs);
            total_valid += valid;
        }
    } else if(solver == MinimalSolver::UPRIGHT) {
	// Solve for projective cameras
        std::vector<Eigen::Matrix<double, 2, 4>> P1, P2, P3, P4;
        int num_projective = upright_radial_quadrifocal_solver(x1, x2, x3, x4, start_system, track_settings, P1, P2, P3, P4, QFs);

        // Upgrade to metric
        for (int i = 0; i < num_projective; ++i) {
            int valid =
                upright_filter_cheirality(x1, x2, x3, x4, P1[i], P2[i], P3[i], P4[i], P1_calib, P2_calib, P3_calib, P4_calib, Xs);
            total_valid += valid;
        }
    } else if(solver == MinimalSolver::NANSON) {
        std::vector<Eigen::Matrix<double, 2, 4>> P1, P2, P3, P4;
        int num_projective = nanson_radial_quadrifocal_solver(x1, x2, x3, x4, start_system, track_settings, P1, P2, P3, P4, QFs);

        for (int i = 0; i < num_projective; ++i) {
            int valid =
                metric_upgrade(x1, x2, x3, x4, P1[i], P2[i], P3[i], P4[i], P1_calib, P2_calib, P3_calib, P4_calib, Xs);
            total_valid += valid;
        }
    } else if(solver == MinimalSolver::NANSON2) {
        std::vector<Eigen::Matrix<double, 2, 4>> P1, P2, P3, P4;
        int num_projective = nanson2_radial_quadrifocal_solver(x1, x2, x3, x4, start_system, track_settings, P1, P2, P3, P4, QFs);

        for (int i = 0; i < num_projective; ++i) {
            int valid =
                metric_upgrade(x1, x2, x3, x4, P1[i], P2[i], P3[i], P4[i], P1_calib, P2_calib, P3_calib, P4_calib, Xs);
            total_valid += valid;
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> runtime_ms = end_time - start_time;

    py::dict result;
    result["P1"] = P1_calib;
    result["P2"] = P2_calib;
    result["P3"] = P3_calib;
    result["P4"] = P4_calib;
    result["Xs"] = Xs;
    result["QFs"] = QFs;
    result["valid"] = total_valid;
    result["runtime"] = runtime_ms.count();

    return result;
}


py::dict radial_quadrifocal_solver_wrapper(const std::vector<Eigen::Vector2d> &x1,
                                    const std::vector<Eigen::Vector2d> &x2,
                                    const std::vector<Eigen::Vector2d> &x3,
                                    const std::vector<Eigen::Vector2d> &x4, const py::dict &opt) {
    TrackSettings track_settings;
    StartSystem start_system;

    if (opt.contains("start_system_file")) {
        std::string filename = opt["start_system_file"].cast<std::string>();
        start_system.load_start_system(filename, MinimalSolver::MINIMAL);
    } else {
        start_system.load_default(MinimalSolver::MINIMAL);
    }

    if (opt.contains("track_settings_file")) {
        std::string filename = opt["track_settings_file"].cast<std::string>();

        track_settings.load_settings(filename);
    } else {
        track_settings = settings_from_dict(opt);
    }

    std::vector<Eigen::Matrix<double, 2, 4>> P1, P2, P3, P4;
    std::vector<Eigen::Matrix<double, 16, 1>> QFs;

    auto start_time = std::chrono::high_resolution_clock::now();

    int valid = radial_quadrifocal_solver(x1, x2, x3, x4, start_system, track_settings, P1, P2, P3, P4, QFs);

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> runtime_ms = end_time - start_time;

    py::dict result;
    result["P1"] = P1;
    result["P2"] = P2;
    result["P3"] = P3;
    result["P4"] = P4;
    result["QFs"] = QFs;
    result["valid"] = valid;
    result["runtime"] = runtime_ms.count();

    return result;
}


py::dict triangulate_wrapper(const std::vector<Eigen::Vector2d> &x1,
                             const std::vector<Eigen::Vector2d> &x2,
                             const std::vector<Eigen::Vector2d> &x3,
                             const std::vector<Eigen::Vector2d> &x4,
			     const Eigen::Matrix<double, 2, 4> &P1,
			     const Eigen::Matrix<double, 2, 4> &P2,
			     const Eigen::Matrix<double, 2, 4> &P3,
			     const Eigen::Matrix<double, 2, 4> &P4)
{
	//std::cout << "triangulate\n";
	//std::cout << P1 << "\n\n";
	rqt::RansacOptions ransac_opt;
	rqt::StartSystem ss;
	rqt::TrackSettings ts;
	rqt::QuadrifocalEstimator qfe(ransac_opt, x1, x2, x3, x4, ss, ts);

	rqt::QuadrifocalEstimator::Reconstruction rec;
	rec.P1 = P1;
	rec.P2 = P2;
	rec.P3 = P3;
	rec.P4 = P4;
	qfe.triangulate(rec);
	//std::cout << rec.X.size() << "\n";
    	py::dict result;
	result["Xs"] = rec.X;
	return result;
}

py::dict metric_upgrade_wrapper(const std::vector<Eigen::Vector2d> &x1, const std::vector<Eigen::Vector2d> &x2,
                                const std::vector<Eigen::Vector2d> &x3, const std::vector<Eigen::Vector2d> &x4,
                                const Eigen::Matrix<double, 2, 4> &P1, const Eigen::Matrix<double, 2, 4> &P2,
                                const Eigen::Matrix<double, 2, 4> &P3, const Eigen::Matrix<double, 2, 4> &P4) {

    std::vector<Eigen::Matrix<double, 2, 4>> P1_calib, P2_calib, P3_calib, P4_calib;
    std::vector<std::vector<Eigen::Vector3d>> Xs;

    int valid = metric_upgrade(x1, x2, x3, x4, P1, P2, P3, P4, P1_calib, P2_calib, P3_calib, P4_calib, Xs);

    py::dict result;
    result["P1"] = P1_calib;
    result["P2"] = P2_calib;
    result["P3"] = P3_calib;
    result["P4"] = P4_calib;
    result["Xs"] = Xs;
    result["valid"] = valid;

    return result;
}

// Helper function for generating C++ code for hard-coding starting systems
void generate_start_system_code(const std::string &filename, const std::string &solver_str) {
    MinimalSolver solver = solver_from_string(solver_str);
    StartSystem start_system;
    start_system.load_start_system(filename, solver);

    std::string problem_name = "default_problem";
    std::string sols_name = "default_sols";

    if(solver == MinimalSolver::UPRIGHT) {
        problem_name = "upright_problem";
        sols_name = "upright_sols";
    }

    std::cout << std::setprecision(16);
    int n = start_system.problem.size();
    std::cout << "std::complex<double> " << problem_name << "[" << n << "] = {";
    for (int i = 0; i < n; ++i) {
        std::cout << "{" << start_system.problem[i].real() << "," << start_system.problem[i].imag() << "}";
        if (i < n - 1) {
            std::cout << ",";
        } else {
            std::cout << "};\n";
        }
    }

    int n1 = start_system.sols.size();
    int n2 = start_system.sols[0].size();
    std::cout << "std::complex<double> " << sols_name << "[" << n1 << "][" << n2 << "] = {\n";
    for (int i = 0; i < n1; ++i) {
        std::cout << "{";
        for (int j = 0; j < n2; ++j) {
            std::cout << "{" << start_system.sols[i][j].real() << "," << start_system.sols[i][j].imag() << "}";
            if (j < n2 - 1) {
                std::cout << ",";
            }
        }
        if (i < n1 - 1) {
            std::cout << "}, // Solution " << i << "\n";
        } else {
            std::cout << "} // Solution " << i << "\n};\n";
        }
    }
}

PYBIND11_MODULE(pyrqt, m) {
    m.doc() = "Python package for estimating radial quadrifocal tensor.";

    /*
    py::class_<CameraPose>(m, "CameraPose")
        .def(py::init<>())
        .def_readwrite("q", &CameraPose::q)
        .def_readwrite("t", &CameraPose::t)
        .def_property("R", &CameraPose::R,
                      [](CameraPose &self, Eigen::Matrix3d R_new) { self.q =
    rotmat_to_quat(R_new); }) .def_property("Rt", &CameraPose::Rt,
                      [](CameraPose &self, Eigen::Matrix<double, 3, 4> Rt_new) {
                          self.q = rotmat_to_quat(Rt_new.leftCols<3>());
                          self.t = Rt_new.col(3);
                      })
        .def("center", &CameraPose::center, "Returns the camera center
    (c=-R^T*t).") .def("E", &CameraPose::E, "Returns the essential matrix E =
    skew(t)*R") .def("__repr__", [](const CameraPose &a) { return "[q: " +
    to_string(a.q.transpose()) + ", " + "t: " + to_string(a.t.transpose()) + "]";
        });
    */

    m.def("radial_quadrifocal_solver", &radial_quadrifocal_solver_wrapper, py::arg("x1"), py::arg("x2"), py::arg("x3"),
          py::arg("x4"), py::arg("options"), "Minimal solver for radial quadrifocal tensor",
          py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>());

    m.def("calibrated_radial_quadrifocal_solver", &calibrated_radial_quadrifocal_solver_wrapper, py::arg("x1"),
          py::arg("x2"), py::arg("x3"), py::arg("x4"), py::arg("options"),
          "Minimal solver for radial quadrifocal tensor",
          py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>());

    m.def("metric_upgrade", &metric_upgrade_wrapper, py::arg("x1"), py::arg("x2"), py::arg("x3"), py::arg("x4"),
          py::arg("P1"), py::arg("P2"), py::arg("P3"), py::arg("P4"), "Upgrades a projective reconstruction to metric.",
          py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>());

   m.def("ransac_quadrifocal", &ransac_quadrifocal_wrapper, py::arg("x1"), py::arg("x2"), py::arg("x3"),
          py::arg("x4"), py::arg("options"), "RANSAC estimator for radial quadrifocal tensor",
          py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>());

   m.def("triangulate", &triangulate_wrapper, py::arg("x1"), py::arg("x2"), py::arg("x3"),
          py::arg("x4"), py::arg("P1"), py::arg("P2"), py::arg("P3"), py::arg("P4"), "Triangulates the lines",
          py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>());

    m.def("generate_start_system_code", &generate_start_system_code, py::arg("filename"), py::arg("solver"),
          "Creates C++ code for hard-coding starting system.",
          py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>());

    m.attr("__version__") = std::string("0.0.2");
}
