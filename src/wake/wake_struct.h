#ifndef WAKE_STRUCT_H
#define WAKE_STRUCT_H

#include <stdio.h>
#include "src/interaction/interaction_utils.h"
#include "src/wake/wake.h"

inline void allocate2DArray(double**& array, const int& row, const int& col, const double& init) {
    array = new double*[row];
    for (size_t i = 0; i < row; i++) {
        array[i] = new double[col];
        for (size_t j = 0; j < col; j++) {
            array[i][j] = init;
        }
    }
}

inline void allocate1DArray(double*& array, const int& len, const double& init) {
    array = new double[len];
    for (size_t i = 0; i < len; i++) {
        array[i] = init;
    }
}

inline void delete2DArray(double** array, const int& row) {
    for (size_t i = 0; i < row; i++) {
        delete[] array[i];
    }
    delete[] array;
}

namespace pawan {
struct wake_struct {
    double _nu = 2.0e-2;
    size_t size;
    size_t numParticles;
    size_t numDimensions = 3;
    double** position;
    double** velocity;
    double** vorticity;
    double** retvorcity;
    double* radius;
    double* volume;
    double* birthstrength;

    wake_struct(pawan::__wake* w) {
        numDimensions = w->_numDimensions;
        numParticles = w->_numParticles;
        size = w->_size;
        allocate2DArray(position, numParticles, numDimensions, 0.0);
        allocate2DArray(velocity, numParticles, numDimensions, 0.0);
        allocate2DArray(vorticity, numParticles, numDimensions, 0.0);
        allocate2DArray(retvorcity, numParticles, numDimensions, 0.0);

        allocate1DArray(radius, numParticles, 1.0);
        allocate1DArray(volume, numParticles, 1.0);
        allocate1DArray(birthstrength, numParticles, 1.0);

        for (size_t i = 0; i < numParticles; i++) {
            radius[i] = gsl_vector_get(w->_radius, i);
            volume[i] = gsl_vector_get(w->_volume, i);
            birthstrength[i] = gsl_vector_get(w->_birthstrength, i);
            for (size_t j = 0; j < numDimensions; j++) {
                position[i][j] = gsl_matrix_get(w->_position, i, j);
                velocity[i][j] = gsl_matrix_get(w->_velocity, i, j);
                vorticity[i][j] = gsl_matrix_get(w->_vorticity, i, j);
                retvorcity[i][j] = gsl_matrix_get(w->_retvorcity, i, j);
            }
        }
    }

    ~wake_struct() {
        delete2DArray(position, numParticles);
        delete2DArray(velocity, numParticles);
        delete2DArray(vorticity, numParticles);
        delete2DArray(retvorcity, numParticles);

        delete[] radius;
        delete[] volume;
        delete[] birthstrength;
    }
};

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