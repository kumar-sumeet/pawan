#include <gsl/gsl_matrix.h>
#include "gsl_utils.h"
#include "src/wake/wake.h"
#include "src/wake/wake_struct.h"

inline void matrixTo2DArray(gsl_matrix* matrix, double** array) {
    for (size_t i = 0; i < matrix->size1; i++) {
        for (size_t j = 0; j < matrix->size2; j++) {
            array[i][j] = gsl_matrix_get(matrix, i, j);
        }
    }
}

inline void vectorToArray(gsl_vector* vector, double* array) {
    for (size_t i = 0; i < vector->size; i++) {
        array[i] = gsl_vector_get(vector, i);
    }
}

inline void delete2DArray(double**& array, int row) {
    for (size_t i = 0; i < row; i++) {
        delete[] array[i];
    }
    delete[] array;
}

inline void wakeToStruct(pawan::__wake* W, pawan::wake_struct* w) {
    w->size = W->_size;
    w->numParticles = W->_numParticles;
    w->numDimensions = W->_numDimensions;
    for (size_t i = 0; i < W->_numParticles; i++) {
        w->radius[i] = gsl_vector_get(W->_radius, i);
        w->volume[i] = gsl_vector_get(W->_volume, i);
        w->birthstrength[i] = gsl_vector_get(W->_birthstrength, i);
        for (size_t j = 0; j < W->_numDimensions; j++) {
            w->position[i][j] = gsl_matrix_get(W->_position, i, j);
            w->velocity[i][j] = gsl_matrix_get(W->_velocity, i, j);
            w->vorticity[i][j] = gsl_matrix_get(W->_vorticity, i, j);
            w->retvorcity[i][j] = gsl_matrix_get(W->_retvorcity, i, j);
        }
    }
}