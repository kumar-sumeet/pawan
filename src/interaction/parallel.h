/*! PArticle Wake ANalysis
 * \file parallel.h
 * \brief Header for OpenMP Parallel Interaction class
 *
 * @author Puneet Singh
 * @date 04/24/2021
 */
#ifndef PARALLEL_H_
#define PARALLEL_H_

#include <gsl/gsl_vector.h>
#include <stdio.h>
#include <iostream>
#include "src/interaction/interaction.h"
#include "src/interaction/interaction_utils.h"
#include "src/utils/print_utils.h"
#include "src/wake/wake.h"

namespace pawan {
class __parallel : public __interaction {
   private:
    //! Interact
    /*
     * Compute interaction between particles of a single wake object
     * \param	W	Wake object pointer
     */
    virtual void interact(__wake* W);

    //! Interact
    /*
     * Compute interaction between particles of two wake objects
     * \param	W1	Wake 1 object pointer
     * \param	W2	Wake 2 object pointer
     */
    virtual void interact(__wake* W1, __wake* W2);

   public:
    //! Constructor
    /*
     * Creates single wake interaction object
     * \param	W	Wake object pointer
     */
    __parallel(__wake* W);

    //! Constructor
    /*
     * Creates empty interaction object with two wakes
     * \param	W1	Wake object pointer
     * \param	W2	Wake object pointer
     */
    __parallel(__wake* W1, __wake* W2);

    //! Destructor
    /*
     * Deletes particles
     */
    ~__parallel() = default;
};
}  // namespace pawan
#endif
