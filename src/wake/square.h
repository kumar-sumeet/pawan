/*! PArticle Wake ANalysis
 * \file square.h
 * \brief Header for square ring class
 *
 * @author Puneet Singh
 * @date 05/13/2021
 */
#ifndef SQUARE_H_
#define SQUARE_H_

#include <gsl/gsl_math.h>
#include <gsl/gsl_matrix.h>
#include <stdio.h>
#include <iostream>

#include "src/wake/wake.h"

namespace pawan {

class __square : public __wake {
   private:
   public:
    //! Constructor for thin vortex ring square
    /*
     * Creates thin vortex square
     * \param gamma		Strength of vortex ring square
     * \param side		Square side length
     * \param core		Core radius
     * \param nRadial	Number of particles/side
     */
    __square(const double& gamma, const double& side, const double& core, const int& nSide);

    //! Destructor
    /*
     * Deletes particles
     */
    ~__square();
};
}  // namespace pawan
#endif
