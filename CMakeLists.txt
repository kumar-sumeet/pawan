cmake_minimum_required (VERSION 3.2)

set(CMAKE_CUDA_ARCHITECTURES "60")
set(CMAKE_CUDA_COMPILER "/HTOpt/cuda/11.6/bin/nvcc") #neccessary for my specific setup TODO remove

project(PaWAn LANGUAGES CXX CUDA)
# add the binary tree to the search path for include files
# so that we will find pawanConfig.h
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)
include_directories("${PROJECT_BINARY_DIR}/config")
include_directories(${PROJECT_SOURCE_DIR})
include_directories(src)

# Other CMakeLists
add_subdirectory(src/wake)
add_subdirectory(src/io)
add_subdirectory(src/system)
add_subdirectory(src/interaction)
add_subdirectory(src/integration)
add_subdirectory(src/resolve)

set(LIB wake io interaction resolve integration system)

find_package( PkgConfig REQUIRED)
pkg_check_modules( gsl REQUIRED gsl )

find_package( OpenMP REQUIRED)

# add the executable
add_executable(pawan src/pawan.cu)#src/test.cu src/test.cuh)
target_link_libraries(pawan ${LIB} ${gsl_LIBRARIES}  ${CMAKE_SOURCE_DIR}/../tecio/teciosrc/bld/libtecio.a)

set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-Ofast -fopenmp -I/../tecio/teciosrc -lpthread  -L/usr/local/lib -lgsl -lgslcblas -lm -std=c++11")
set(CUDAFLAGS ${CUDAFLAGS} " fopenmp")

# Required standards in C++
set(needed_features
	cxx_constexpr
	cxx_range_for)

target_compile_features(pawan PUBLIC ${needed_features})
