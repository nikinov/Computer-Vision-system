# CMakeList.txt : CMake project for TorchTest, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)

project ("TorchTest")

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS})

# Add source to this project's executable.
add_executable (TorchTest "TorchTest.cpp" "TorchTest.h")
target_link_libraries(TorchTest "${TORCH_LIBRARIES}")
target_link_libraries(TorchTest ${OpenCV_LIBS})

if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET TorchTest
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:TorchTest>)
endif (MSVC)

# TODO: Add tests and install targets if needed.
