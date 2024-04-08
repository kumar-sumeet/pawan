/*! PArticle Wake ANalysis
 * \file vring.h
 * \brief Header for Vortex Ring example class
 *
 * @author Puneet Singh
 * @date 08/12/2021
 */
#ifndef VRING_H_
#define VRING_H_

#include <stdio.h>
#include <iostream>
#include <gsl/gsl_math.h>
#include <gsl/gsl_matrix.h>

#include "src/wake/wake.h"
#include "src/wake/ring.h"

namespace pawan{

class __vring : public __ring{

	private:
		double _R;	/*!< Ring radius */
		double _sigmaR;	/*!< Ring core radius */
		size_t _nAzi;	/*!< Number of azimuthal sections */
		size_t _nLayers;/*!< Number of layers in the cross-section */
		double _dPsi;	/*!< Azimuthal step size */
		gsl_vector *_strength;	/*!< Strengths of particles n a layer */
		
		//! Set vorticity
		/*
		 * Set vortucity assuming axial symmetry
		 */
		virtual void set_vorticity();

	protected:
		
		//! Create particles
		/*
		 * Creates empty particles
		 * \param n	Number of particles
		 */
		virtual void create_particles(const int &n);
        //! Add particles once private members set
		//virtual void addParticles();

	public:
		//! Constructor for vogel model vortex ring
		/*
		 * Creates vortex ring
		 * \param radius	Ring radius
		 * \param core		Core radius
		 * \param nLayers	Number of core layers
		 * \param nAzi	    Azimuthal discretization
		 */
		__vring(const double &radius,
                const double &core,
                const int &nLayers,
                const int &nAzi,
                const double &sig);
		
		//! Destructor
		/*
		 * Deletes particles
		 */
		~__vring();

		//! Print all wake particles
		/*
		 * Print wake particle information
		 */
		virtual void print();

		//! Write wake data file
		/*
		 * Write binary file with all wake particle data
		 * \param f	Binary file
		 */
		//virtual void save(FILE *f);

		//! Write wake data file
		/*
		 * Write binary file with all wake particle data
		 * \param f	Binary file
		 */
		//virtual void write(FILE *f);

		//! Read wake data file
		/*
		 * Read binary file with all wake particle data
		 * \param op	Input/Output object for file
		 */
		//virtual void read(__io *IO);
		
		//! Set states
		/*
		 * Sets positions and vorticities of particles
		 * \param state		Wake state
		 */
		virtual void setStates(const gsl_vector *state);

		//! Get rates
		/*
		 * Get velocities and rate of change of vorticities of particles
		 * \param rate		Wake rate
		 */
		virtual void getRates(gsl_vector *rate);
		
		//! Get states
		/*
		 * Get position and vorticities of particles
		 * \param rate		Wake state
		 */
		virtual void getStates(gsl_vector *state);
		
		//! Get desired rates
		/*
		 * Get ideal wake rates
		 * \param rate		Wake rate
		 */
		virtual void getIdealRates(gsl_vector *rate);
};
}
#endif
