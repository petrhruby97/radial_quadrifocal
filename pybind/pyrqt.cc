#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/iostream.h>
#include <map>
#include "rqt/types.h"
#include "rqt/radial_quadrifocal_solver.h"
#include "rqt/metric_upgrade.h"


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

/*
py::dict dict_from_ransac_stats(const RansacStats &stats) {
    py::dict result;
    result["refinements"] = stats.refinements;
    result["iterations"] = stats.iterations;
    result["num_inliers"] = stats.num_inliers;
    result["inlier_ratio"] = stats.inlier_ratio;
    result["model_score"] = stats.model_score;

    return result;
}

Camera camera_from_dict(const py::dict &camera_dict) {
    Camera camera;
    camera.model_id = Camera::id_from_string(camera_dict["model"].cast<std::string>());

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

py::dict radial_quadrifocal_solver_wrapper(const std::vector<Eigen::Vector2d> &x1, 
                                           const std::vector<Eigen::Vector2d> &x2, 
                                           const std::vector<Eigen::Vector2d> &x3, 
                                           const std::vector<Eigen::Vector2d> &x4,
                                           const py::dict &track_settings_dict) {
    std::string data_file = "starting_system.txt";
    std::vector<std::complex<double>> problem;
    std::vector<std::vector<std::complex<double>>> sols;
    rqt::load_start_system(data_file, problem, sols);

    std::cout << "Loaded starting system (" << problem.size() << " coeffs, " << sols.size() << " solutions)" << std::endl;
    
    TrackSettings track_settings = settings_from_dict(track_settings_dict);

    std::vector<Eigen::Matrix<double, 2, 4>> P1, P2, P3, P4;
    std::vector<Eigen::Matrix<double, 16, 1>> QFs;

    std::cout << "Entering solver" << std::endl;
    int valid = radial_quadrifocal_solver(problem, sols, x1, x2, x3, x4, track_settings, P1, P2, P3, P4, QFs);

    py::dict result;
    result["P1"] = P1;
    result["P2"] = P2;
    result["P3"] = P3;
    result["P4"] = P4;    
    result["QFs"] = QFs;    
    result["valid"] = valid;

    return result;
}


py::dict metric_upgrade_wrapper(const std::vector<Eigen::Vector2d> &x1, 
                                const std::vector<Eigen::Vector2d> &x2, 
                                const std::vector<Eigen::Vector2d> &x3, 
                                const std::vector<Eigen::Vector2d> &x4,
                                const Eigen::Matrix<double,2,4> &P1, 
                                const Eigen::Matrix<double,2,4> &P2, 
                                const Eigen::Matrix<double,2,4> &P3, 
                                const Eigen::Matrix<double,2,4> &P4) {

    std::vector<Eigen::Matrix<double, 2, 4>> P1_calib, P2_calib, P3_calib, P4_calib;    
    std::vector<std::vector<Eigen::Vector3d>> Xs;

    int valid = metric_upgrade(x1,x2,x3,x4,P1,P2,P3,P4,P1_calib,P2_calib,P3_calib,P4_calib,Xs);

    py::dict result;
    result["P1"] = P1_calib;
    result["P2"] = P2_calib;
    result["P3"] = P3_calib;
    result["P4"] = P4_calib;    
    result["Xs"] = Xs;    
    result["valid"] = valid;

    return result;
}

PYBIND11_MODULE(pyrqt, m)
{
    m.doc() = "Python package for estimating radial quadrifocal tensor.";



    /*
    py::class_<CameraPose>(m, "CameraPose")
        .def(py::init<>())
        .def_readwrite("q", &CameraPose::q)
        .def_readwrite("t", &CameraPose::t)
        .def_property("R", &CameraPose::R,
                      [](CameraPose &self, Eigen::Matrix3d R_new) { self.q = rotmat_to_quat(R_new); })
        .def_property("Rt", &CameraPose::Rt,
                      [](CameraPose &self, Eigen::Matrix<double, 3, 4> Rt_new) {
                          self.q = rotmat_to_quat(Rt_new.leftCols<3>());
                          self.t = Rt_new.col(3);
                      })
        .def("center", &CameraPose::center, "Returns the camera center (c=-R^T*t).")
        .def("E", &CameraPose::E, "Returns the essential matrix E = skew(t)*R")
        .def("__repr__", [](const CameraPose &a) {
            return "[q: " + to_string(a.q.transpose()) + ", " + "t: " + to_string(a.t.transpose()) + "]";
        });
    */

    m.def("radial_quadrifocal_solver", &radial_quadrifocal_solver_wrapper, py::arg("x1"), py::arg("x2"), py::arg("x3"),
          py::arg("x4"), py::arg("track_settings"), "Minimal solver for radial quadrifocal tensor", py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>());

    m.def("metric_upgrade", &metric_upgrade_wrapper, py::arg("x1"), py::arg("x2"), py::arg("x3"),
          py::arg("x4"), py::arg("P1"), py::arg("P2"), py::arg("P3"), py::arg("P4"), "Upgrades a projective reconstruction to metric.", py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>());

    m.attr("__version__") = std::string("0.0.1");
  
}
