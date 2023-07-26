add_library(wake SHARED wake.cpp ring.cpp square.cpp vring.cpp test_wake.cpp)
target_compile_features(wake PUBLIC cxx_constexpr)
