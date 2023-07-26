add_library(resolve SHARED resolve.cpp)
target_compile_features(resolve PUBLIC cxx_constexpr)