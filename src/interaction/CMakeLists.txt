add_library(interaction SHARED bnh.cpp bnh.cu bnh_node.cu bnh_particle.cu bnh_utility.cu gsl_memory.cu interaction.cpp parallel.cpp)
target_compile_features(interaction PUBLIC cxx_constexpr)
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-fopenmp -lpthread -O3 -lgsl -lm")
set_target_properties(interaction PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
