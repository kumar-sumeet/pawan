add_library(integration SHARED integration.cpp rk4.cpp gpu_euler.cu gpu_euler.cuh)
target_compile_features(integration PUBLIC cxx_constexpr)
