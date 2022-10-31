set(SOURCES
    radial_quadrifocal_solver.cc
    linear_radial_quadrifocal_solver.cc
    upright_radial_quadrifocal_solver.cc
    nanson_radial_quadrifocal_solver.cc
    nanson2_radial_quadrifocal_solver.cc
    quadrifocal_estimator.cc
    metric_upgrade.cc
    upright_filter_cheirality.cc
    solver_det4.cc
    types.cc
)


# Set HEADERS_PUBLIC variable
set(HEADERS_PUBLIC    
    radial_quadrifocal_solver.h
    linear_radial_quadrifocal_solver.h
    upright_radial_quadrifocal_solver.h
    nanson_radial_quadrifocal_solver.h
    nanson2_radial_quadrifocal_solver.h
    quadrifocal_estimator.h
    metric_upgrade.h
    upright_filter_cheirality.h
    types.h
    ransac_impl.h
)

# library configuration
include(${PROJECT_SOURCE_DIR}/cmake/LibraryConfig.cmake)
target_link_libraries(rqt PUBLIC Eigen3::Eigen ${CERES_LIBRARIES})
target_compile_features(rqt PRIVATE cxx_std_17)
