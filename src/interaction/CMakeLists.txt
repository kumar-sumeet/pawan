# Add the cuda source file to the library sources
set(INTERACTION_SOURCES interaction.cpp parallel.cpp interaction_utils_cuda.cu)

# Compile step_cuda_wrapper.cu using nvcc
find_package(CUDA REQUIRED)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-O3")
cuda_add_library(interaction SHARED ${INTERACTION_SOURCES})
target_compile_features(interaction PUBLIC cxx_constexpr)

# Link with external libraries
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-fopenmp -lpthread -O3 -lgsl -lm")
target_link_libraries(interaction ${CUDA_LIBRARIES} ${CUDA_cusparse_LIBRARY})
