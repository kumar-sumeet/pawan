/*! PArticle Wake ANalysis
 * \file rk4.h
 * \brief Header for Runge Kutta 4th order integration class
 *
 * @author Puneet Singh
 * @date 04/29/2021
 */
#ifndef RK4_H_
#define RK4_H_

#include "src/integration/integration.h"

namespace pawan {
class __rk4 : public __integration {
   protected:
    //! Time step
    /*
     * Advance one time step
     * \param	dt	Time step
     * \param	S	Interaction solver
     * \param	state	System state
     */
    virtual void step(const double& dt, __interaction* S, gsl_vector* state);
    virtual void step(const double& dt, wake_struct* W, double* states, const int& len);

   public:
    //! Constructor
    /*
     * Creates empty integral
     * \param	t	Total time
     * \param	n	Number of time steps
     */
    __rk4(const double& t, const size_t& n);

    //! Destructor
    /*
     * Deletes particles
     */
    ~__rk4() = default;
};
}  // namespace pawan
#endif
