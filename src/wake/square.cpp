/*! PArticle Wake ANalysis
 * \file square.cpp
 * \brief Routines for creating Vortex squares
 *
 * @author Puneet Singh
 * @date 05/13/2021
 */
#include "square.h"

pawan::__square::__square(const double& gamma, const double& side, const double& core, const int& nSide) {
    create_particles(4 * nSide);
    double volume = 4.0 * M_PI * gsl_pow_3(core) / 3.0; /*!< Volume of particle */
    double length = side / nSide;                       /*!< Length of particle */
    double strength = gamma * length;                   /*!< Strength of particle */
    double dist[4] = {0.5 * side, 0.5 * side, -0.5 * side, -0.5 * side};
    double vort[4] = {strength, -strength, -strength, strength};
    for (size_t i = 0; i < 4; ++i) {
        size_t axisPerp = i % 2;
        size_t axisParl = 1 - axisPerp;
        double sign = pow(-1, (i / 2) % 2);
        for (size_t j = 0; j < nSide; ++j) {
            size_t index = i * nSide + j;
            gsl_matrix_set(_position, index, axisPerp, dist[i]);
            gsl_matrix_set(_position, index, axisParl, sign * (-0.5 * side + length * (j + 0.5)));
            gsl_matrix_set(_position, index, 2, 0.0);
            gsl_matrix_set(_vorticity, index, axisPerp, 0.0);
            gsl_matrix_set(_vorticity, index, axisParl, vort[i]);
            gsl_matrix_set(_vorticity, index, 2, 0.0);
            gsl_vector_set(_radius, index, core);
            gsl_vector_set(_volume, index, volume);
            gsl_vector_set(_birthstrength, index, strength);
        }
    }
}

pawan::__square::~__square() {
}
