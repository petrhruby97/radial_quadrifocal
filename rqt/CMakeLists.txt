set(SOURCES
    radial_quadrifocal_solver.cc
    metric_upgrade.cc
    solver_det4.cc
)


# Set HEADERS_PUBLIC variable
set(HEADERS_PUBLIC    
radial_quadrifocal_solver.h
metric_upgrade.h
)

# library configuration
include(${PROJECT_SOURCE_DIR}/cmake/LibraryConfig.cmake)
target_link_libraries(rqt PUBLIC Eigen3::Eigen ${CERES_LIBRARIES})
target_compile_features(rqt PRIVATE cxx_std_17)