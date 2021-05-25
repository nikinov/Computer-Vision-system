# CMakeList.txt : CMake project for Predictor, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)

project ("Predictor")

find_package(Torch REQUIRED)

# Add source to this project's executable.
add_executable (Predictor "Predictor.cpp" "Predictor.h")
target_link_libraries(Predictor "${TORCH_LIBRARIES}")

if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET Predictor
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:Predictor>)
endif (MSVC)

# TODO: Add tests and install targets if needed.
