# CMakeList.txt : CMake project for Predictor, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.18.1)

project (Predictor)


set(CMAKE_PREFIX_PATH "${PROJECT_SOURCE_DIR}/lib/libtorch/share/cmake/Torch")

find_package(torch REQUIRED)
find_package(OpenCV REQUIRED)
include_directories(&{OpenCV_INCLUDE_DIRS})

# Add source to this project's executable.

#add_executable (${PROJECT_NAME} "src/Predictor.cpp" "headers/Predictor.h" )

add_library(${PROJECT_NAME}
  STATIC
  "src/Predictor.cpp" "headers/Predictor.h")

target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}")
target_link_libraries(${PROJECT_NAME} "${OpenCV_LIBS}")

if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET Predictor
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:Predictor>)
endif (MSVC)

# TODO: Add tests and install targets if needed.
