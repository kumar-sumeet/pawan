add_library(interaction SHARED interaction.cpp parallel.cpp gpu.cu gpu.cuh)
target_compile_features(interaction PUBLIC cxx_constexpr)
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-fopenmp -lpthread -O3 -lgsl -lm")
