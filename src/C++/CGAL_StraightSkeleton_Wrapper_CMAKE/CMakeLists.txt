# CMakeList.txt : 
# project specific logic here.
#
cmake_minimum_required (VERSION 3.18.1)

project (CGAL_StraightSkeleton_Wrapper)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

find_package(CGAL CONFIG REQUIRED)
find_package(Boost REQUIRED)

find_package(OpenMP)

set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")

add_library(${PROJECT_NAME}
  SHARED
  "src/dllmain.cpp" "src/CGAL_StraightSkeleton_Wrapper.cpp" "src/stdafx.cpp"
  "headers/framework.h" "headers/stdafx.h" "headers/targetver.h")

target_link_libraries(${PROJECT_NAME} PRIVATE CGAL::CGAL)
target_link_libraries(${PROJECT_NAME} PRIVATE Boost::boost)

# TODO: Add tests and install targets if needed.
