#ifndef WAKE_STRUCT_H
#define WAKE_STRUCT_H

#include <stdio.h>
#include "src/interaction/interaction_utils.h"

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

    wake_struct(const int& n) {
        numDimensions = 3;
        numParticles = n;
        size = 2 * numParticles * numDimensions;
        allocate2DArray(position, numParticles, numDimensions, 0.0);
        allocate2DArray(velocity, numParticles, numDimensions, 0.0);
        allocate2DArray(vorticity, numParticles, numDimensions, 0.0);
        allocate2DArray(retvorcity, numParticles, numDimensions, 0.0);

        allocate1DArray(radius, numParticles, 1.0);
        allocate1DArray(volume, numParticles, 1.0);
        allocate1DArray(birthstrength, numParticles, 1.0);
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

    void setStates(const double* state) {
        size_t np = size / 2 / numDimensions;
        size_t matrixsize = np * numDimensions;

        for (size_t i = 0; i < np; i++) {
            for (size_t j = 0; j < numDimensions; j++) {
                position[i][j] = state[i * numDimensions + j];
            }
        }

        for (size_t i = 0; i < np; i++) {
            for (size_t j = 0; j < numDimensions; j++) {
                vorticity[i][j] = state[i * numDimensions + j + matrixsize];
            }
        }
    }

    void getStates(double* state) {
        for (size_t i = 0; i < numParticles; i++) {
            for (size_t j = 0; j < numDimensions; j++) {
                size_t ind = i * numDimensions + j;
                state[ind] = position[i][j];
                ind += size / 2;
                state[ind] = vorticity[i][j];
            }
        }
    }

    void getRates(double* rate) {
        for (size_t i = 0; i < numParticles; i++) {
            for (size_t j = 0; j < numDimensions; j++) {
                size_t ind = i * numDimensions + j;
                rate[ind] = velocity[i][j];
                ind += size / 2;
                rate[ind] = retvorcity[i][j];
            }
        }
    }

    void interact() {
        for (size_t i = 0; i < numParticles; i++)
            for (size_t j = 0; j < numDimensions; j++)
                velocity[i][j] = retvorcity[i][j] = 0;

        for (size_t i_src = 0; i_src < numParticles; i_src++) {
            double* r_src = new double[numDimensions];
            double* a_src = new double[numDimensions];
            double* dr_src = new double[numDimensions];
            double* da_src = new double[numDimensions];
            for (size_t j = 0; j < numDimensions; j++) {
                r_src[j] = position[i_src][j];
                a_src[j] = vorticity[i_src][j];
                dr_src[j] = velocity[i_src][j];
                da_src[j] = retvorcity[i_src][j];
            }
            double s_src = radius[i_src];
            double v_src = volume[i_src];
            for (size_t i_trg = i_src + 1; i_trg < numParticles; i_trg++) {
                double* r_trg = new double[numDimensions];
                double* a_trg = new double[numDimensions];
                double* dr_trg = new double[numDimensions];
                double* da_trg = new double[numDimensions];
                for (size_t j = 0; j < numDimensions; j++) {
                    r_trg[j] = position[i_trg][j];
                    a_trg[j] = vorticity[i_trg][j];
                    dr_trg[j] = velocity[i_trg][j];
                    da_trg[j] = retvorcity[i_trg][j];
                }
                double s_trg = radius[i_trg];
                double v_trg = volume[i_trg];

                INTERACT_CUDA(_nu, s_src, s_trg, r_src, r_trg, a_src, a_trg, v_src, v_trg, dr_src, dr_trg, da_src, da_trg);
            }
        }
    }
};
}  // namespace pawan

#endif  // WAKE_STRUCT_H