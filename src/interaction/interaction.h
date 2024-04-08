/*! PArticle Wake ANalysis
 * \file interaction.h
 * \brief Header for Interaction class
 *
 * @author Puneet Singh
 * @date 04/24/2021
 */
#ifndef INTERACTION_H_
#define INTERACTION_H_

#include <stdio.h>
#include <iostream>
#include <vector>

#include <gsl/gsl_vector.h>
#include "src/utils/print_utils.h"
#include "src/system/system.h"
#include "src/wake/wake.h"
#include "src/networkinterface/networkdatastructures.h"

namespace pawan{
class __interaction : public __system{

	protected:
		double _nu;			/*!< Kinematic viscosity */
		size_t _nWake;			/*!< Number of wake objects*/
		gsl_vector *_totalVorticity;	/*!< Total vorticity */
		gsl_vector *_linearImpulse;	/*!< Linear Impulse */
		gsl_vector *_angularImpulse;	/*!< Angular Impulse */
		double _enstrophy;		/*!< Enstrophy */
		double _helicity;		/*!< Helicity */
		double _kineticEnergy;		/*!< Kinetic Energy */
        double _enstrophyF;		/*!< div-free Enstrophy */
        double _kineticEnergyF;		/*!< div-free Kinetic Energy */
        double _Zc=0.0;        /*!< position of (vring) wake centroid */
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
		
		//! Interact
		/*
		 * Compute interaction between wake particles
		 */
		virtual void interact();
		
		//! Influence
		/*
		 * Compute vorticity field
		 */
		virtual void influence();

	private:
		std::vector<pawan::__wake *> _W;
		
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

	public:
		//size_t _size;		[>!< Size of state vector <]
		
		//! Constructor
		/*
		 * Creates empty interaction object with no wake
		 */
		__interaction();
		
		//! Constructor
		/*
		 * Creates empty interaction object with one wake
		 * \param	W	Wake object pointer
		 */
		__interaction(__wake *W);
		
		//! Constructor
		/*
		 * Creates empty interaction object with two wakes
		 * \param	W1	Wake object pointer
		 * \param	W2	Wake object pointer
		 */
		__interaction(__wake *W1, __wake *W2);
		
		//! Destructor
		/*
		 * Deletes particles
		 */
		~__interaction();
		
		//! Diagnose
		/*
		 * Compute all diagnostic terms for the flow field
		 */
		virtual void diagnose();

		//! Add wake structure
		/*
		 * Add wake object
		 */
		virtual void addWake(__wake *W);

		//! Solve system
		/*
		 * Solve system
		 */
		virtual void solve();

		//! Resolve system
		/*
		 * Resolve system
		 */
		virtual void resolve();

		//! Write all wake data
		/*
		 * Write binary file with all wake particle data
		 * \param f	Binary file
		 */
		virtual void write(FILE *f);
        //write diagnostics to file
        virtual void writediagnosis(FILE *fdiag);
        virtual void writenu(FILE *fdiag);

		//! Set states
		/*
		 * Sets all wake states
		 * \param state		Wake state
		 */
		virtual void setStates(const gsl_vector *state);
		
		//! Get rates
		/*
		 * Get wake state rates
		 * \param rate		Wake rate
		 */
		virtual void getRates(gsl_vector *rate);
		
		//! Get states
		/*
		 * Get current wake states
		 * \param state		Wake state
		 */
		virtual void getStates(gsl_vector *state);

        //! add relaxation to wake system
        virtual void relax(size_t &stepnum);
		//! add new particles
		virtual void addParticles(PawanRecvData pawanrecvdata,size_t &stepnum);
        //! translate particles with Vinf
        virtual void updateVinfEffect(const double *Vinf,double &dt);
        //! translate particles due to induced vel from bound vortices
        virtual void updateBoundVorEffect(PawanRecvData pawanrecvdata,double &dt);
        //! get induced due to all wake particles at each airstation
        virtual void getInflow(PawanRecvData pawanrecvdata, PawanSendData pawansenddata);
        //! get velocity induced due to all wake particles at a given location
        virtual void getVi(const gsl_vector *r, gsl_vector *vi,const size_t &n);//n is dummy, remove alter
		//! Get ideal rates
		/*
		 * Get ideal rates
		 * \param rate		Wake rate
		 */
		virtual void getIdealRates(gsl_vector *rate);
};
}
#endif
