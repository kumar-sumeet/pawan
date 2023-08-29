/*! PArticle Wake ANalysis
 * \file gaussian.h
 * \brief Inline functions for Gaussian smoothing
 *
 * @author 	Puneet Singh
 * @date	08/20/2021
 *
 */

#ifndef GAUSSIAN_H_
#define GAUSSIAN_H_

#include "src/utils/gsl_utils.h"

/*! \fn inline double ZETASIG(const double &rho, const double &sigma, double &q, double &F, double &Z)
 * \brief Compute vortex particle kernel
 * \param	rho		double distance
 * \param	sigma		double radius
 */
inline double ZETASIG(	const double &rho, const double &sigma){
	double rho_bar = rho/sigma;
	return exp(-0.5*rho_bar*rho_bar)/pow(sigma,3.0)/pow(2.0*M_PI,1.5);
};

/*! \fn inline double QSIG(const double &rho, const double &sigma, double &q)
 * \brief Compute velocity induced by vortex particle kernel
 * \param	rho		double distance
 * \param	sigma		double radius
 */
inline double QSIG(	const double &rho, 
			const double &sigma){
	double rho_bar = rho/sigma;
	double sig3 = sigma*sigma*sigma;
	double Z = ZETASIG(rho,sigma);
	double phi = 0.25*M_1_PI*erf(M_SQRT1_2*rho_bar)/sig3;
	return (phi/rho_bar - Z)/gsl_pow_2(rho_bar);
};

/*! \fn inline void KERNEL(const double &rho, const double &sigma, double &q, double &F, double &Z)
 * \brief Compute kernel functions 
 * \param	rho		double distance
 * \param	sigma		double radius
 * \param	q		double Q
 * \param	F		double F
 * \param	Z		double Z
 */
inline void KERNEL(	const double &rho, 
			const double &sigma, 
			double &q, 
			double &F, 
			double &Z,
			double &n){
	double rho_bar = rho/sigma;
	double sig3 = sigma*sigma*sigma;
	Z = ZETASIG(rho,sigma);
	double phi = 0.25*M_1_PI*erf(M_SQRT1_2*rho_bar)/sig3;
	q = (phi/rho_bar - Z)/gsl_pow_2(rho_bar);
	n = ZETASIG(rho,sigma);
	F = (Z - 3.0*q)/gsl_pow_2(rho);
};

/*! \fn inline double ENST(const double &rho, const double &sigma, double &q)
 * \brief Compute enstrophy induced by vortex particle kernel
 * \param	rho		double distance
 * \param	sigma		double radius
 * \param	F1		double F1 factor for a1.a2
 * \param	F2		double F2 factor for (a1.x12)(a2.x12)
 */
inline void ENST(	const double &rho, 
			const double &sigma,
			double &F1,
			double &F2){
	F1 = 0.0;
	F2 = 0.0;
};

inline void ENSTF(	const double &rho,
                      const double &sigma,
                      double &F1){
    F1 = 0.0;
};

/*! \fn inline double ENST(const double &rho, const double &sigma, double &q)
 * \brief Compute enstrophy induced by vortex particle kernel
 * \param	sigma		double radius
 */
inline double ENST(	const double &sigma){
	return 0.0;
};
inline double ENSTF(	const double &sigma){
    return 0.0;
};


#endif
