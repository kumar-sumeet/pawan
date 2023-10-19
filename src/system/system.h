/*! PArticle Wake ANalysis
 * \file system.h
 * \brief Header for System class
 *
 * @author Puneet Singh
 * @date 06/24/2021
 */
#ifndef SYSTEM_H_
#define SYSTEM_H_

#include <stdio.h>
#include <iostream>
#include <vector>

#include <gsl/gsl_vector.h>
#include "src/utils/print_utils.h"
#include "src/networkinterface/networkdatastructures.h"
#include "../teciosrc/TECIO.h"

namespace pawan{
class __system{

	public:
		size_t _size;		/*!< Size of state vector */
		size_t _totalmaxsize=0;/*!< total max permissible size of state vector of all wakes */
		//! Constructor
		/*
		 * Creates empty system object
		 */
		__system();
		
		//! Destructor
		/*
		 * Delete system
		 */
		~__system() = default;

		//! Solvel system 
		/*
		 * Solve system
		 */
		virtual void solve(){};

		//! Write all system data
		/*
		 * Write binary file with all system data
		 * \param f	Binary file
		 */
		virtual void write(FILE *f){};
		//write diagnostics to file
        virtual void writediagnosis(FILE *fdiag){};
        virtual void printdiagnostics(double *totalDiag){};
        virtual void writenu(FILE *fdiag){};

		//! Set states
		/*
		 * Sets all system states
		 * \param state		System state
		 */
		virtual void setStates(const gsl_vector *state){};
		
		//! Get rates
		/*
		 * Get system rates
		 * \param rate		System rate
		 */
		virtual void getRates(gsl_vector *rate){};
		
		//! Get states
		/*
		 * Get current System states
		 * \param state		System state
		 */
		virtual void getStates(gsl_vector *state){};

        /*!
         * copy particles from buffer for GPU to the wakes
         * pointer should be actually double4
         */
        virtual void setParticles(double *particles) = 0;
        /*!
         * copy particles from the wakes to a buffer for GPU
         * pointer should be actually double4
         */
        virtual void getParticles(double *particles) = 0;
        virtual void getParticles_arr(double *particles) = 0;
        virtual int amountParticles() = 0;
        virtual int totalmaxParticles() = 0;
        virtual double getNu() = 0;
        virtual void setDiagnostics(double *totalDiag) = 0;



        //! remesh the wake system
        virtual void split(size_t &stepnum){};
        virtual void merge(size_t &stepnum){};
        //! add new particles
        virtual void addParticles(PawanRecvData pawanrecvdata,size_t &stepnum){};
       //! translate particles with Vinf
        virtual void updateVinfEffect(const double *Vinf,double &dt){};
        //! translate particles due to induced vel from bound vortices
        virtual void updateBoundVorEffect(PawanRecvData pawanrecvdata,double &dt){};
        //! get induced due to all wake particles at each airstation
        virtual void getInflow(PawanRecvData pawanrecvdata, PawanSendData pawansenddata){};
        //! Diagnose
        /*
         * Compute all diagnostic terms for the flow field
         */
        virtual void diagnose(){};

};
}
#endif
