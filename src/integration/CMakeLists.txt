add_library(integration SHARED integration.cpp rk4.cpp rk3.h rk3.cpp gpu_int.cu gpu_int.cuh)
target_compile_features(integration PUBLIC cxx_constexpr)
