/*! PArticle Wake ANalysis
 * \file rk4.cpp
 * \brief Routines for Runge-Kutta 4th order integrations
 *
 * @author Puneet Singh
 * @date 04/29/2021
 */
#include "rk4.h"

pawan::__rk4::__rk4(const double& t, const size_t& n)
    : __integration(t, n) {}

void pawan::__rk4::step(const double& dt, __interaction* S, gsl_vector* states) {
    gsl_vector* x1 = gsl_vector_calloc(states->size);
    gsl_vector* x2 = gsl_vector_calloc(states->size);
    gsl_vector* x3 = gsl_vector_calloc(states->size);
    gsl_vector* k1 = gsl_vector_calloc(states->size);
    gsl_vector* k2 = gsl_vector_calloc(states->size);
    gsl_vector* k3 = gsl_vector_calloc(states->size);
    gsl_vector* k4 = gsl_vector_calloc(states->size);

    gsl_vector_memcpy(x1, states);

    // k1 = f(x,t)
    S->setStates(states);
    S->interact();
    S->getRates(k1);

    // x1 = x + 0.5*dt*k1
    gsl_vector_memcpy(x1, k1);
    gsl_vector_scale(x1, 0.5 * dt);
    gsl_vector_add(x1, states);

    // k2 = f(x1,t+0.5*dt)
    S->setStates(x1);
    S->interact();
    S->getRates(k2);

    // x2 = x1 + 0.5*dt*dx2
    gsl_vector_memcpy(x2, k2);
    gsl_vector_scale(x2, 0.5 * dt);
    gsl_vector_add(x2, states);

    // k3 = f(x2,t+0.5*dt)
    S->setStates(x2);
    S->interact();
    S->getRates(k3);

    // x3 = x2 + dt*k3
    gsl_vector_memcpy(x3, k3);
    gsl_vector_scale(x3, dt);
    gsl_vector_add(x3, states);

    // k4 = f(x3,t+dt)
    S->setStates(x3);
    S->interact();
    S->getRates(k4);

    gsl_vector_add(k1, k4);
    gsl_vector_scale(k1, dt / 6.);

    gsl_vector_add(k2, k3);
    gsl_vector_scale(k2, dt / 3.);

    gsl_vector_add(k1, k2);

    gsl_vector_add(states, k1);

    S->setStates(states);

    gsl_vector_free(x1);
    gsl_vector_free(x2);
    gsl_vector_free(x3);
    gsl_vector_free(k1);
    gsl_vector_free(k2);
    gsl_vector_free(k3);
    gsl_vector_free(k4);
}

void pawan::__rk4::step(const double& dt, wake_struct* w, double* states, const int& len) {
    double *x1, *x2, *x3, *k1, *k2, *k3, *k4;
    allocate1DArray(x1, len, 0);
    allocate1DArray(x2, len, 0);
    allocate1DArray(x3, len, 0);
    allocate1DArray(k1, len, 0);
    allocate1DArray(k2, len, 0);
    allocate1DArray(k3, len, 0);
    allocate1DArray(k4, len, 0);

    for (size_t i = 0; i < len; i++)
        x1[i] = states[i];

    // k1 = f(x,t)
    w->setStates(states);
    w->interact();
    w->getRates(k1);

    // x1 = x + 0.5*dt*k1
    for (size_t i = 0; i < len; i++) {
        x1[i] = k1[i];
        x1[i] *= 0.5 * dt;
        x1[i] += states[i];
    }

    // k2 = f(x1, t+0.5*dt)
    w->setStates(x1);
    w->interact();
    w->getRates(k2);

    // x2 = x1 + 0.5*dt*dx2
    for (size_t i = 0; i < len; i++) {
        x2[i] = k2[i];
        x2[i] *= 0.5 * dt;
        x2[i] += states[i];
    }

    // k3 = f(x2, t+0.5*dt)
    w->setStates(x2);
    w->interact();
    w->getRates(k3);

    // x3 = x2 + dt*k3
    for (size_t i = 0; i < len; i++) {
        x3[i] = k3[i];
        x3[i] *= 0.5 * dt;
        x3[i] += states[i];
    }

    // k4 = f(x3, t+dt)
    w->setStates(x3);
    w->interact();
    w->getRates(k4);

    for (size_t i = 0; i < len; i++) {
        k1[i] += k4[i];
        k1[i] *= dt / 6.;

        k2[i] += k3[i];
        k2[i] *= dt / 3.;

        k1[i] += k2[i];

        states[i] += k1[i];
    }
    w->setStates(states);

    delete[] x1;
    delete[] x2;
    delete[] x3;
    delete[] k1;
    delete[] k2;
    delete[] k3;
    delete[] k4;
}