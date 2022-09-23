set(SOURCES
    radial_quadrifocal_solver.cc
    quadrifocal_estimator.cc
    metric_upgrade.cc
    solver_det4.cc
    types.cc
)


# Set HEADERS_PUBLIC variable
set(HEADERS_PUBLIC    
    radial_quadrifocal_solver.h
    quadrifocal_estimator.h
    metric_upgrade.h
    types.h
    ransac_impl.h
)

# library configuration
include(${PROJECT_SOURCE_DIR}/cmake/LibraryConfig.cmake)
target_link_libraries(rqt PUBLIC Eigen3::Eigen ${CERES_LIBRARIES})
target_compile_features(rqt PRIVATE cxx_std_17)