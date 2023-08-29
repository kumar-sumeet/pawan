#include "test_wake.h"

namespace pawan {

//create a random wake
    test_wake::test_wake(int size, gsl_rng * r){

        create_particles(size);

        for(int i = 0; i < size; i++){
            for(int j = 0; j < 3; j++ )
                gsl_matrix_set(_position,i,j,gsl_rng_uniform(r) * 6 -3);
        }

        for(int i = 0; i < size; i++){
            for(int j = 0; j < 3; j++ )
                gsl_matrix_set(_vorticity,i,j,gsl_rng_uniform(r) * 6 -3);
        }

        for(int i = 0; i < size; i++){
            gsl_vector_set(_radius,i,gsl_rng_uniform(r));

        }

        for(int i = 0; i < size; i++){
            gsl_vector_set(_volume,i,gsl_rng_uniform(r));

        }

    }

}
