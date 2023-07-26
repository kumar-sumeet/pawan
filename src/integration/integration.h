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
#include "src/system/system.h"
#include "src/io/io.h"
#include "src/networkinterface/networkdatastructures.h"
#include "src/networkinterface/networkinterface.h"
#include "src/networkinterface/networkinterface.cpp" //templates included this way

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
        virtual void step(const double &dt,__system *S,
                          gsl_vector *state);

	public:
		//! Constructor
		/*
		 * Creates empty integral
		 * \param	t	Total time
		 * \param	n	Number of time steps
		 */
		__integration(const double &t, const size_t &n);
        __integration();

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
        virtual void integrate(__system *S,
                       __io *IO,
                       NetworkInterfaceTCP<OPawanRecvData,OPawanSendData> *networkCommunicatorTest,
                       bool diagnose=false);
        virtual void integrate(__system *S,
                       __io *IO,
                       bool diagnose=false);

};
}
#endif
