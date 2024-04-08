cmake_minimum_required (VERSION 3.2)
project(PaWAn LANGUAGES CXX)
# add the binary tree to the search path for include files
# so that we will find pawanConfig.h

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
add_executable(pawan src/pawan.cpp)
target_link_libraries(pawan ${LIB} ${gsl_LIBRARIES})

set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-Ofast -fopenmp -lpthread -L/usr/local/lib -lgsl -lgslcblas -lm -std=c++11")

# Required standards in C++
set(needed_features
	cxx_constexpr
	cxx_range_for)

target_compile_features(pawan PUBLIC ${needed_features})
