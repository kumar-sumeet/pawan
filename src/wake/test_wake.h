#pragma once

#include "wake.h"

#include <gsl/gsl_rng.h>

namespace pawan {

class test_wake : public __wake {

public:
//create a wake with n random particles
test_wake(int size, gsl_rng * r);
};


}