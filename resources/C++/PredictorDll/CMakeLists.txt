cmake_minimum_required(VERSION 3.18)

project(Predictor_dll)

# add sources
file(GLOB DLL_CPP "*.cpp")
file(GLOB DLL_H   "*.h")
set(DLL_SOURCES ${DLL_CPP} ${DLL_H})

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)
include_directories(&{OpenCV_INCLUDE_DIRS})

# generate dll library from the sources
add_library(${PROJECT_NAME}
	SHARED
	${DLL_SOURCES})

# add external libraries
target_link_libraries(${PROJECT_NAME}
	Predictor
)

# add link directories
target_link_directories(${PROJECT_NAME}
    PRIVATE
)

# add include directories
target_include_directories(${PROJECT_NAME}
	PRIVATE
	${PATH_PREDICTOR}
)

add_custom_command(TARGET ${PROJECT_NAME}
		POST_BUILD
		COMMAND ${CMAKE_COMMAND} -E copy
				"${PATH_PREDICTOR}/lib/libtorch/lib/torch_cuda_cu.dll">
				"${CMAKE_CURRENT_BINARY_DIR}/$<$<CONFIG:debug>:Debug>$<$<CONFIG:release>:Release>")


#add_custom_command(TARGET ${PROJECT_NAME}
#		POST_BUILD
#		COMMAND ${CMAKE_COMMAND} -E copy
#				"${PATH_PREDICTOR}/lib/libtorch/lib/torch_cuda_cpp.dll">
#				"${CMAKE_CURRENT_BINARY_DIR}/$<$<CONFIG:debug>:Debug>$<$<CONFIG:release>:Release>")