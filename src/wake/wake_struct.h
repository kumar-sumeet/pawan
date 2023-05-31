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

inline void setStates(wake_struct* w, const double* state) {
    size_t numDimensions = w->numDimensions;
    size_t np = w->size / 2 / numDimensions;
    size_t matrixsize = np * numDimensions;

    for (size_t i = 0; i < np; i++) {
        for (size_t j = 0; j < numDimensions; j++) {
            w->position[i][j] = state[i * numDimensions + j];
        }
    }

    for (size_t i = 0; i < np; i++) {
        for (size_t j = 0; j < numDimensions; j++) {
            w->vorticity[i][j] = state[i * numDimensions + j + matrixsize];
        }
    }
}

inline void getStates(wake_struct* w, double* state) {
    size_t numDimensions = w->numDimensions;
    size_t numParticles = w->numParticles;
    for (size_t i = 0; i < numParticles; i++) {
        for (size_t j = 0; j < numDimensions; j++) {
            size_t ind = i * numDimensions + j;
            state[ind] = w->position[i][j];
            ind += w->size / 2;
            state[ind] = w->vorticity[i][j];
        }
    }
}

inline void getRates(wake_struct* w, double* rate) {
    size_t numDimensions = w->numDimensions;
    size_t numParticles = w->numParticles;
    for (size_t i = 0; i < numParticles; i++) {
        for (size_t j = 0; j < numDimensions; j++) {
            size_t ind = i * numDimensions + j;
            rate[ind] = w->velocity[i][j];
            ind += w->size / 2;
            rate[ind] = w->retvorcity[i][j];
        }
    }
}

inline void interact(wake_struct* w) {
    size_t numDimensions = w->numDimensions;
    size_t numParticles = w->numParticles;
    for (size_t i = 0; i < numParticles; i++)
        for (size_t j = 0; j < numDimensions; j++)
            w->velocity[i][j] = w->retvorcity[i][j] = 0.0;

    for (size_t i_src = 0; i_src < numParticles; i_src++) {
        const double* r_src = &(w->position[i_src][0]);
        const double* a_src = &(w->vorticity[i_src][0]);
        double* dr_src = &(w->velocity[i_src][0]);
        double* da_src = &(w->retvorcity[i_src][0]);
        double s_src = w->radius[i_src];
        double v_src = w->volume[i_src];
        for (size_t i_trg = i_src + 1; i_trg < numParticles; i_trg++) {
            const double* r_trg = &(w->position[i_trg][0]);
            const double* a_trg = &(w->vorticity[i_trg][0]);
            double* dr_trg = &(w->velocity[i_trg][0]);
            double* da_trg = &(w->retvorcity[i_trg][0]);
            double s_trg = w->radius[i_trg];
            double v_trg = w->volume[i_trg];

            INTERACT_GSL_FREE(w->_nu, s_src, s_trg, r_src, r_trg, a_src, a_trg, v_src, v_trg, dr_src, dr_trg, da_src, da_trg);
        }
    }
}
}  // namespace pawan

#endif  // WAKE_STRUCT_H