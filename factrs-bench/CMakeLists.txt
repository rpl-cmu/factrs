cmake_minimum_required(VERSION 3.16)

project(factrs-benchmark-deps
    VERSION 0.1.0
    LANGUAGES CXX
)
include(FetchContent)

# ------------------------- Include Eigen ------------------------- #
# set(BUILD_TESTING OFF CACHE BOOL "")
# set(EIGEN_BUILD_TESTING OFF CACHE BOOL "")
# set(EIGEN_BUILD_PKGCONFIG OFF CACHE BOOL "")
# set(EIGEN_BUILD_DOC OFF CACHE BOOL "")
# FetchContent_Declare(
#     Eigen3
#     GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
#     GIT_TAG        3.4.0
#     GIT_SHALLOW TRUE
#     GIT_PROGRESS   TRUE
#     USES_TERMINAL_DOWNLOAD TRUE
#     OVERRIDE_FIND_PACKAGE
# )
# FetchContent_MakeAvailable(Eigen3)

# ------------------------- Include GTSAM ------------------------- #
# Enable features to make closer to factrs default
set(GTSAM_USE_QUATERNIONS ON CACHE BOOL "")
set(GTSAM_SLOW_BUT_CORRECT_BETWEENFACTOR ON CACHE BOOL "")
set(GTSAM_SLOW_BUT_CORRECT_EXPMAP ON CACHE BOOL "")
set(GTSAM_BUILD_WITH_MARCH_NATIVE ON CACHE BOOL "")
set(BUILD_SHARED_LIBS OFF CACHE BOOL "")
set(GTSAM_BUILD_TYPE_POSTFIXES OFF CACHE BOOL "")
# Turn off extra building
set(GTSAM_BUILD_TESTS OFF CACHE BOOL "")
set(GTSAM_BUILD_DOCS OFF CACHE BOOL "")
set(GTSAM_BUILD_PYTHON OFF CACHE BOOL "")
set(GTSAM_INSTALL_MATLAB_TOOLBOX OFF CACHE BOOL "")
set(GTSAM_BUILD_UNSTABLE OFF CACHE BOOL "")
set(GTSAM_BUILD_EXAMPLES OFF CACHE BOOL "")
set(GTSAM_BUILD_EXAMPLES_ALWAYS OFF CACHE BOOL "")
# set(GTSAM_USE_SYSTEM_EIGEN ON CACHE BOOL "")

FetchContent_Declare(
    GTSAM
    GIT_REPOSITORY https://github.com/borglab/gtsam.git
    GIT_TAG        4.2.0
    GIT_SHALLOW TRUE
    GIT_PROGRESS   TRUE
    USES_TERMINAL_DOWNLOAD TRUE
)
FetchContent_MakeAvailable(GTSAM)

option(BENCH_BUILD_BENCH OFF)
if(BENCH_BUILD_BENCH)
    add_library(depends_bench src/gtsam.cpp)
    target_link_libraries(depends_bench PUBLIC gtsam)
    target_include_directories(depends_bench PUBLIC ../)
endif()

install(TARGETS gtsam DESTINATION .)