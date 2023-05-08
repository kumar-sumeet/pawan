/*! PArticle Wake ANalysis
 * \file integration.cpp
 * \brief Routines for Integrations
 *
 * @author Puneet Singh
 * @date 04/21/2021
 */
#include "integration.h"

extern "C" void cuda_step_wrapper(const double _dt, pawan::wake_struct* w, const double* state_array);

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

    pawan::__wake* wake = S->getWake();
    // for (size_t i = 0; i < wake->_numParticles; i++) {
    //     std::cout << gsl_matrix_get(wake->_position, i, 0) << " " << gsl_matrix_get(wake->_position, i, 1) << " " << gsl_matrix_get(wake->_position, i, 2) << " " << std::endl;
    // }

    double tStart = TIME();
    for (size_t i = 1; i <= 1; ++i) {
        OUT("\tStep", i);
        t = i * _dt;
        step(_dt, S, states);
        fwrite(&t, sizeof(double), 1, f);
        S->write(f);
    }
    fclose(f);
    double tEnd = TIME();
    OUT("Total Time (s)", tEnd - tStart);

    for (size_t i = 0; i < wake->_numParticles; i++) {
        std::cout << gsl_matrix_get(wake->_position, i, 0) << " " << gsl_matrix_get(wake->_position, i, 1) << " " << gsl_matrix_get(wake->_position, i, 2) << " " << std::endl;
    }

    gsl_vector_free(states);
}

void pawan::__integration::integrate_cuda(__interaction* S) {
    gsl_vector* states = gsl_vector_calloc(S->_size);
    S->getStates(states);
    double* state_array = new double[S->_size];
    vectorToArray(states, state_array);
    pawan::__wake* wake = S->getWake();
    wake_struct* w = new wake_struct(wake);
    cuda_step_wrapper(_dt, w, state_array);
    delete w;
    delete[] state_array;
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