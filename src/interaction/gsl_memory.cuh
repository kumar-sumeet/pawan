#pragma once

#include <memory>

#include <gsl/gsl_vector.h>

struct gsl_vector_delete
{
    void operator()(gsl_vector* const vector) const
    {
        gsl_vector_free(vector);
    }
};

using gsl_unique_vector = std::unique_ptr<gsl_vector, gsl_vector_delete>;

gsl_unique_vector gsl_make_unique_vector_3();
gsl_unique_vector gsl_make_unique_vector_3(double3);
