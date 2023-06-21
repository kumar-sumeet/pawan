#ifndef WAKE_STRUCT_H
#define WAKE_STRUCT_H

#include <stdio.h>
#include "src/interaction/interaction_utils.h"
#include "src/wake/wake.h"

namespace pawan {
struct wake_cuda {
    double _nu = 2.0e-2;
    size_t size;
    size_t numParticles;
    size_t numDimensions = 3;
    double* position;
    double* velocity;
    double* vorticity;
    double* retvorcity;
    double* radius;
    double* volume;
    double* birthstrength;
};

}  // namespace pawan

#endif  // WAKE_STRUCT_H