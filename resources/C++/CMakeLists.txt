cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(predict VERSION 0.1.0)

# We want C++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MD")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MDd")

set(PATH_PREDICTOR "${PROJECT_SOURCE_DIR}/Predictor")

set(CMAKE_PREFIX_PATH "${PROJECT_SOURCE_DIR}/Predictor/lib/libtorch/share/cmake/Torch")

add_subdirectory(Predictor)
add_subdirectory(PredictorDll)