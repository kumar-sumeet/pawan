/*! PArticle Wake ANalysis
 * \file integration.cpp
 * \brief Routines for Integrations
 *
 * @author Puneet Singh
 * @date 04/21/2021
 */
#include "integration.h"

extern "C" void step_cuda_1(const double dt, pawan::wake_struct* w, double* states, const int len);

pawan::__integration::__integration(const double& t, const size_t& n) {
    _dt = t / n;
    _t = t;
    _n = n;
}

void pawan::__integration::integrate(__interaction* S, __io* IO) {
    gsl_vector* states = gsl_vector_calloc(S->_size);
    FILE* f = IO->create_binary_file(".wake");
    double t = 0.0;
    fwrite(&t, sizeof(double), 1, f);
    S->write(f);
    S->getStates(states);
    double tStart = TIME();
    for (size_t i = 1; i <= 2; ++i) {
        OUT("\tStep", i);
        t = i * _dt;
        step(_dt, S, states);
        fwrite(&t, sizeof(double), 1, f);
        S->write(f);
    }
    fclose(f);
    double tEnd = TIME();
    pawan::__wake* wake = S->getWake();
    for (size_t i = 0; i < wake->_numParticles; i++) {
        std::cout << gsl_matrix_get(wake->_position, i, 0) << " " << gsl_matrix_get(wake->_position, i, 1) << " " << gsl_matrix_get(wake->_position, i, 2) << " " << std::endl;
    }
    OUT("Total Time (s)", tEnd - tStart);
    gsl_vector_free(states);
}

void pawan::__integration::integrate_cuda(__interaction* S) {
    gsl_vector* states = gsl_vector_calloc(S->_size);
    S->getStates(states);
    double* state_array = new double[S->_size];
    vectorToArray(states, state_array);
    pawan::__wake* wake = S->getWake();
    wake_struct* w = new wake_struct(wake);
    double tStart = TIME();
    for (size_t i = 1; i <= 2; i++) {
        OUT("\tStep", i);
        step_cuda_1(_dt, w, state_array, S->_size);  // cuda version step.
    }
    double tEnd = TIME();
    for (size_t i = 0; i < w->numParticles; i++) {
        std::cout << w->position[i][0] << " " << w->position[i][1] << " " << w->position[i][2] << " " << std::endl;
    }
    OUT("Total Time (s)", tEnd - tStart);
    delete w;
    delete [] state_array;
    gsl_vector_free(states);
}

void pawan::__integration::step(const double& dt, __interaction* S, gsl_vector* states) {
    gsl_vector* rates = gsl_vector_calloc(S->_size);
    S->interact();
    S->getRates(rates);
    gsl_vector_scale(rates, dt);
    gsl_vector_add(states, rates);
    S->setStates(states);
    gsl_vector_free(rates);
}