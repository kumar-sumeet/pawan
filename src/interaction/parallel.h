/*! PArticle Wake ANalysis
 * \file parallel.h
 * \brief Header for OpenMP Parallel Interaction class
 *
 * @author Puneet Singh
 * @date 04/24/2021
 */
#ifndef PARALLEL_H_
#define PARALLEL_H_

#include <stdio.h>
#include <iostream>
#include <gsl/gsl_vector.h>
#include "src/utils/print_utils.h"
#include "src/wake/wake.h"
#include "src/interaction/interaction.h"
#include "src/interaction/interaction_utils.h"

namespace pawan{
class __parallel : public __interaction{

	private:
		
		//! Calculate Kinetic Energy
		/*
		 * Calculates kinetic energy of the wake
		 * \param	W	Wake object pointer
		 */
		virtual double calculateKineticEnergy(__wake *W);
        //divergence-free KE
        virtual double calculateKineticEnergyF(__wake *W);

		//! Calculate Kinetic Energy
		/*
		 * Calculates kinetic energy of the wake
		 * \param	W1	Wake 1 object pointer
		 * \param	W2	Wake 2 object pointer
		 */
		virtual double calculateKineticEnergy(__wake *W1, __wake *W2);
        //divergence-free KE
        virtual double calculateKineticEnergyF(__wake *W1, __wake *W2);

		//! Interact
		/*
		 * Compute interaction between particles of a single wake object
		 * \param	W	Wake object pointer
		 */
		virtual void interact(__wake *W);
		
		//! Interact
		/*
		 * Compute interaction between particles of two wake objects
		 * \param	W1	Wake 1 object pointer
		 * \param	W2	Wake 2 object pointer
		 */
		virtual void interact(__wake *W1, __wake *W2);
		
		//! Influence
		/*
		 * Computes vorticity field of particles of a single wake object
		 * \param	W	Wake object pointer
		 */
		virtual void influence(__wake *W);
		
		//! Influence
		/*
		 * Computes vorticity field of particles of two wake objects
		 * \param	W1	Wake 1 object pointer
		 * \param	W2	Wake 2 object pointer
		 */
		virtual void influence(__wake *W1, __wake *W2);
		
		//! Calculate Angular Impulse
		/*
		 * Calculates angular impulse of the wake
		 * \param	W	Wake object pointer
		 * \param	A	Angular impulse vector
		 */
		virtual void calculateAngularImpulse(__wake *W, gsl_vector *A);
		
		//! Calculate Linear Impulse
		/*
		 * Calculates linear impulse of the wake
		 * \param	W	Wake object pointer
		 * \param	I	Linear impulse vector
		 */
		virtual void calculateLinearImpulse(__wake *W, gsl_vector *I);
		
		//! Calculate Total Vorticity
		/*
		 * Calculates total vorticity of the wake
		 * \param	W	Wake object pointer
		 * \param	O	Total vorticity vector
		 */
		virtual void calculateTotalVorticity(__wake *W, gsl_vector *O);
		
		//! Calculate Enstrophy
		/*
		 * Calculates enstrophy of the wake
		 * \param	W	Wake object pointer
		 */
		virtual double calculateEnstrophy(__wake *W);
        //divergence-free enstrophy
        virtual double calculateEnstrophyF(__wake *W);

		//! Calculate Enstrophy
		/*
		 * Calculates enstrophy of the wake
		 * \param	W1	Wake 1 object pointer
		 * \param	W2	Wake 2 object pointer
		 */
		virtual double calculateEnstrophy(__wake *W1, __wake *W2);
        //divergence-free enstrophy
        virtual double calculateEnstrophyF(__wake *W1, __wake *W2);

		//! Calculate Helicity
		/*
		 * Calculates helicity of the wake
		 * \param	W	Wake object pointer
		 */
		virtual double calculateHelicity(__wake *W);
		
		//! Calculate Helicity
		/*
		 * Calculates helicity of the wake
		 * \param	W1	Wake 1 object pointer
		 * \param	W2	Wake 2 object pointer
		 */
		virtual double calculateHelicity(__wake *W1, __wake *W2);

	public:
		
		//! Constructor
		/*
		 * Creates single wake interaction object
		 * \param	W	Wake object pointer
		 */
		__parallel(__wake *W);
		
		//! Constructor
		/*
		 * Creates empty interaction object with two wakes
		 * \param	W1	Wake object pointer
		 * \param	W2	Wake object pointer
		 */
		__parallel(__wake *W1, __wake *W2);
		
		//! Destructor
		/*
		 * Deletes particles
		 */
		~__parallel() = default;
};
}
#endif
