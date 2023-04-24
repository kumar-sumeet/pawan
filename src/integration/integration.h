/*! PArticle Wake ANalysis
 * \file integration.h
 * \brief Header for Integration class
 *
 * @author Puneet Singh
 * @date 04/21/2021
 */
#ifndef INTEGRATION_H_
#define INTEGRATION_H_

#include <stdio.h>
#include <iostream>
#include <gsl/gsl_vector.h>
#include "src/utils/print_utils.h"
#include "src/utils/timing_utils.h"
#include "src/interaction/interaction.h"
#include "src/io/io.h"
#include "src/wake/wake_struct.h"

namespace pawan{
class __integration{

	protected:
		double _dt;	/*!< Time step size */
		double _t;	/*!< Total time */
		size_t _n;	/*!< Number of time steps */
		
		//! Time step
		/*
		 * Advance one time step
		 * \param	dt	Time step
		 * \param	S	Interaction solver
		 * \param	state	System state
		 */
		virtual void step(const double &dt,__interaction *S, gsl_vector *state);
		virtual void step(const double &dt, wake_struct *W, double* states) = 0;

	public:
		//! Constructor
		/*
		 * Creates empty integral
		 * \param	t	Total time
		 * \param	n	Number of time steps
		 */
		__integration(const double &t, const size_t &n);
		
		//! Destructor
		/*
		 * Deletes particles
		 */
		~__integration() = default;
		
		//! Integrate
		/*
		 * Integrates wake
		 * \param	S	Interaction solver
		 * \param	IO	Input/Output file writing
		 */
		void integrate(__interaction *S, __io *IO);

		void integrate_cuda(__interaction *S);
};
}
#endif
