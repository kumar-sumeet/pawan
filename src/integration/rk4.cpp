/*! PArticle Wake ANalysis
 * \file rk4.cpp
 * \brief Routines for Runge-Kutta 4th order integrations
 *
 * @author Puneet Singh
 * @date 04/29/2021
 */
#include "rk4.h"

pawan::__rk4::__rk4(const double &t, const size_t &n):__integration(t,n){}

void pawan::__rk4::step(const double &dt, __interaction *S, gsl_vector* states){
	gsl_vector *x1 = gsl_vector_calloc(states->size);
	gsl_vector *x2 = gsl_vector_calloc(states->size);
	gsl_vector *x3 = gsl_vector_calloc(states->size);
	gsl_vector *k1 = gsl_vector_calloc(states->size);
	gsl_vector *k2 = gsl_vector_calloc(states->size);
	gsl_vector *k3 = gsl_vector_calloc(states->size);
	gsl_vector *k4 = gsl_vector_calloc(states->size);

	gsl_vector_memcpy(x1,states);
	
	// k1 = f(x,t)
	S->setStates(states);
	S->interact();
	S->getRates(k1);
	
	// x1 = x + 0.5*dt*k1
	gsl_vector_memcpy(x1,k1);
	gsl_vector_scale(x1,0.5*dt);
	gsl_vector_add(x1,states);
	
	// k2 = f(x1,t+0.5*dt)
	S->setStates(x1);
	S->interact();
	S->getRates(k2);
	
	// x2 = x1 + 0.5*dt*dx2
	gsl_vector_memcpy(x2,k2);
	gsl_vector_scale(x2,0.5*dt);
	gsl_vector_add(x2,states);
	
	// k3 = f(x2,t+0.5*dt)
	S->setStates(x2);
	S->interact();
	S->getRates(k3);
	
	// x3 = x2 + dt*k3
	gsl_vector_memcpy(x3,k3);
	gsl_vector_scale(x3,dt);
	gsl_vector_add(x3,states);
	
	// k4 = f(x3,t+dt)
	S->setStates(x3);
	S->interact();
	S->getRates(k4);

	gsl_vector_add(k1,k4);
	gsl_vector_scale(k1,dt/6.);
	
	gsl_vector_add(k2,k3);
	gsl_vector_scale(k2,dt/3.);

	gsl_vector_add(k1,k2);
	
	gsl_vector_add(states,k1);

	S->setStates(states);

	gsl_vector_free(x1);
	gsl_vector_free(x2);
	gsl_vector_free(x3);
	gsl_vector_free(k1);
	gsl_vector_free(k2);
	gsl_vector_free(k3);
	gsl_vector_free(k4);
}


void pawan::__rk4::step(const double &dt, wake_struct *W, double* states){
	std::cout << "test" << dt << std::endl;
}