# CMakeList.txt : CMake project for Predictor, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.18.1)

project (Predictor_Dll)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

set(Torch_DIR "C:/libtorch/share/cmake/Torch")

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

find_package(OpenMP)

set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")

include_directories(&{OpenCV_INCLUDE_DIRS})

# Add source to this project's executable.

#add_executable (${PROJECT_NAME} "src/Predictor.cpp" "headers/Predictor.h" )

add_library(${PROJECT_NAME}
  SHARED
  "src/dllmain.cpp" "headers/dllmain.h")

target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}")
target_link_libraries(${PROJECT_NAME} "${OpenCV_LIBS}")

if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET Predictor_Dll
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:Predictor_Dll>)
endif (MSVC)

# TODO: Add tests and install targets if needed.
