/*! PArticle Wake ANalysis
 * \file rk3.h
 * \brief Header for Runge Kutta 3rd order integration class
 *
 * @author Sumeet Kumar
 * @date 07/03/2023
 */
#ifndef RK3_H
#define RK3_H

#include "src/integration/integration.h"

namespace pawan{
    class __rk3 : public __integration{

    protected:

        //! Time step
        /*
         * Advance one time step
         * \param	dt	Time step
         * \param	S	Interaction solver
         * \param	state	System state
         */
        virtual void step(const double &dt,__system *S, gsl_vector *state);

    public:
        //! Constructor
        /*
         * Creates empty integral
         * \param	t	Total time
         * \param	n	Number of time steps
         */
        __rk3(const double &t, const size_t &n);

        //! Destructor
        /*
         * Deletes particles
         */
        ~__rk3() = default;
    };
}

#endif //RK3_H
