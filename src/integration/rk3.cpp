/*! PArticle Wake ANalysis
 * \file rk3.cpp
 * \brief Routines for Runge-Kutta 3rd order integrations
 *
 * @author Sumeet Kumar
 * @date 07/03/2023
 */
#include "rk3.h"

pawan::__rk3::__rk3(const double &t, const size_t &n):__integration(t,n){}

void pawan::__rk3::step(const double &dt, __system *S, gsl_vector* states){
    gsl_vector *x1 = gsl_vector_calloc(states->size);
    gsl_vector *x2 = gsl_vector_calloc(states->size);
    gsl_vector *k1 = gsl_vector_calloc(states->size);
    gsl_vector *k2 = gsl_vector_calloc(states->size);
    gsl_vector *k3 = gsl_vector_calloc(states->size);

    gsl_vector_memcpy(x1,states);

    // k1 = f(x,t)
    S->setStates(states);
    S->solve();
    S->getRates(k1);

    // x1 = x + 0.5*dt*k1
    gsl_vector_memcpy(x1,k1);
    gsl_vector_scale(x1,0.5*dt);
    gsl_vector_add(x1,states);

    // k2 = f(x1,t+0.5*dt)
    S->setStates(x1);
    S->solve();
    S->getRates(k2);

    // x2 = x -dt*k1 + 2*dt*k2
    gsl_vector_memcpy(x2,k2);
    gsl_vector_scale(x2,2.*dt);
    gsl_vector_add(x2,states);
    gsl_vector_scale(k1,dt);
    gsl_vector_sub(x2,k1);

    // k3 = f(x2,t+dt)
    S->setStates(x2);
    S->solve();
    S->getRates(k3);

    gsl_vector_scale(k3,dt/6.0);
    gsl_vector_scale(k1,1.0/6.0);
    gsl_vector_add(k1,k3);

    gsl_vector_scale(k2,dt*2.0/3.0);

    gsl_vector_add(k1,k2);

    gsl_vector_add(states,k1);

    S->setStates(states);

    gsl_vector_free(x1);
    gsl_vector_free(x2);
    gsl_vector_free(k1);
    gsl_vector_free(k2);
    gsl_vector_free(k3);
}
