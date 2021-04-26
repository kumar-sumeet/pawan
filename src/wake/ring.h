/*! PArticle Wake ANalysis
 * \file ring.h
 * \brief Header for Ring class
 *
 * @author Puneet Singh
 * @date 04/24/2021
 */
#ifndef RING_H_
#define RING_H_

#include <stdio.h>
#include <iostream>
#include <gsl/gsl_math.h>
#include <gsl/gsl_matrix.h>

#include "src/wake/wake.h"

namespace pawan{

class __ring : public __wake{

	private:

	public:
		
		//! Constructor for thin vortex ring
		/*
		 * Creates thin vortex ring
		 * \param gamma		Strength of ring	
		 * \param radius	Ring radius
		 * \param core		Core radius
		 * \param nRadial	Number of particles
		 */
		__ring(const double &gamma, const double &radius, const double &core, const int &nRadial);
		
		//! Constructor for vogel model vortex ring
		/*
		 * Creates thin vortex ring
		 * \param gamma		Strength of ring	
		 * \param radius	Ring radius
		 * \param core		Core radius
		 * \param nRadial	Number of radial sections
		 * \param nCore		Number of core layers
		 */
		__ring(const double &gamma, const double &radius, const double &core, const int &nRadial, const int &nCore);
		
		//! Destructor
		/*
		 * Deletes particles
		 */
		~__ring();
};
}
#endif
