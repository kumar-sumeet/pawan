/*! PArticle Wake ANalysis
 * \file integration.cpp
 * \brief Routines for Integrations
 *
 * @author Puneet Singh
 * @date 04/21/2021
 */
#include "integration.h"

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
    for (size_t i = 1; i <= _n; ++i) {
        OUT("\tStep", i);
        t = i * _dt;
        step(_dt, S, states);
        fwrite(&t, sizeof(double), 1, f);
        S->write(f);
    }
    fclose(f);
    double tEnd = TIME();
    OUT("Total Time (s)", tEnd - tStart);
    gsl_vector_free(states);
}

void pawan::__integration::integrate_cuda(__interaction* S) {
    gsl_vector* states = gsl_vector_calloc(S->_size);
    S->getStates(states);
    double* state_array = new double[S->_size];
    vectorToArray(states, state_array);
    pawan::__wake* wake = S->getWake();
    wake_struct* w = new wake_struct(S->_size / 6);
    wakeToStruct(wake, w);
    double tStart = TIME();
    for (size_t i = 1; i <= _n; i++) {
        OUT("\tStep", i);
        step(_dt, w, state_array, S->_size);  // cuda version step.
    }
    double tEnd = TIME();
    OUT("Total Time (s)", tEnd - tStart);
    delete w;
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