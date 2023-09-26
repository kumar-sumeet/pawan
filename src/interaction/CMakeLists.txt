add_library(interaction SHARED interaction.cu parallel.cpp gpu.cu gpu.cuh gpu_common.cu gpu_common.cuh gaussian.cuh highorder.cuh)
target_compile_features(interaction PUBLIC cxx_constexpr)
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-fopenmp -lpthread -O3 -lgsl -lm")
