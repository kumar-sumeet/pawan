/*! PArticle Wake ANalysis
 * @file timing_utils.h
 * @brief Timing utilities for PAWAN
 *
 * @author Puneet Singh
 * @date 04/28/2021
 *
 */

#ifndef TIMING_UTILS_H_
#define TIMING_UTILS_H_

#include <omp.h>
#include <sys/time.h>
#include <string>

/*! \fn inline double TIME()
 * \brief Return time
 */
inline double TIME() {
    return omp_get_wtime();
};

#endif
