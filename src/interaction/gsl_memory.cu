#include "interaction/gsl_memory.cuh"

gsl_unique_vector gsl_make_unique_vector_3()
{
    return gsl_unique_vector(gsl_vector_calloc(3));
}
gsl_unique_vector gsl_make_unique_vector_3(double3 const value)
{
    auto& vector = *gsl_vector_alloc(3);
    gsl_vector_set(&vector, 0, value.x);
    gsl_vector_set(&vector, 1, value.y);
    gsl_vector_set(&vector, 2, value.z);

    return gsl_unique_vector(&vector);
}
