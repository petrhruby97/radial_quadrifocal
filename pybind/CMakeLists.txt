# setup PYTHON_EXECUTABLE

find_package (Python COMPONENTS Interpreter Development)
message(STATUS "Python_EXECUTABLE: " ${Python_EXECUTABLE})

add_subdirectory(pybind11)

pybind11_add_module(pyrqt MODULE pyrqt.cc)
target_link_libraries(pyrqt PUBLIC rqt Eigen3::Eigen ${CERES_LIBRARIES})
target_compile_features(pyrqt PRIVATE cxx_std_17)
