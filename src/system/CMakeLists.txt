add_library(system SHARED system.cpp )
target_compile_features(system PUBLIC cxx_constexpr)
