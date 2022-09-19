cmake_minimum_required(VERSION 3.10)

project(rqt VERSION 0.0.1)
set(CMAKE_FIND_LIBRARY_SUFFIXES .a)
set(CMAKE_CXX_STANDARD 17)

# Set variables
set(LIBRARY_NAME   ${PROJECT_NAME})
set(LIBRARY_FOLDER ${PROJECT_NAME})
include(${PROJECT_SOURCE_DIR}/cmake/SetEnv.cmake)

option(ASAN_BUILD "Build with address sanitizer" OFF)
if (ASAN_BUILD)
	add_compile_options("-fsanitize=address")
	link_libraries("-fsanitize=address")
endif ()

# Eigen
find_package(Eigen3 3.4 REQUIRED)
find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

# Library sources
add_subdirectory(rqt)

# python bindings
add_subdirectory(pybind)


# Compilation options
if(MSVC)
	target_compile_options(${LIBRARY_NAME} PRIVATE /bigobj /fp:fast)
else()
	target_compile_options(${LIBRARY_NAME} PRIVATE
		-march=native -Wall -Wno-sign-compare -Wno-unused-variable -fPIC)
endif()

