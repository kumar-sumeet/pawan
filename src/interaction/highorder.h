/*! PArticle Wake ANalysis
 * \file highorder.h
 * \brief Inline functions for High Order Algebraic smoothing
 *
 * @author 	Puneet Singh
 * @date	08/20/2021
 *
 */

#ifndef HIGHORDER_H_
#define HIGHORDER_H_

#include "src/utils/gsl_utils.h"

/*! \fn inline double ZETASIG(const double &rho, const double &sigma, double &q, double &F, double &Z)
 * \brief Compute vortex particle kernel
 * \param	rho		double distance
 * \param	sigma		double radius
 */
inline double ZETASIG(	const double &rho, const double &sigma){
    //DOUT("----------------ZETASIG()--------------");
	double rho_bar = rho/sigma;
	return 1.875*M_1_PI/pow(rho_bar*rho_bar + 1.0,3.5)/pow(sigma,3);
};

/*! \fn inline double QSIG(const double &rho, const double &sigma, double &q)
 * \brief Compute velocity induced by vortex particle kernel
 * \param	rho		double distance
 * \param	sigma		double radius
 */
inline double QSIG(	const double &rho, 
			const double &sigma){
    //DOUT("----------------QSIG()--------------");
	double rho_bar = rho/sigma;
	double rho_bar2 = gsl_pow_2(rho_bar);
    double rho3 = rho*rho*rho;
	return 0.25*M_1_PI*rho_bar2*rho_bar*(rho_bar2 + 2.5)/pow(rho_bar2 + 1.0,2.5)/rho3;
};

inline double ETASIG(const double &rho, const double &sigma){
    double rho_bar = rho/sigma;
    return 13.125*M_1_PI/pow(rho_bar*rho_bar + 1.0,4.5)/pow(sigma,3);
};

/*! \fn inline void KERNEL(const double &rho, const double &sigma, double &q, double &F, double &Z)
 * \brief Compute velocity induced by vortex particle kernel
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
    //DOUT("----------------KERNEL()--------------");
	Z = ZETASIG(rho,sigma);
	q = QSIG(rho,sigma);
	n = ETASIG(rho,sigma);
	F = (Z - 3.0*q)/gsl_pow_2(rho);
};

/*! \fn inline void ENST(const double &rho, const double &sigma, double &q)
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
    //DOUT("----------------ENST()--------------");
	double rho2 = gsl_pow_2(rho);
	double sig2 = gsl_pow_2(sigma);
	double factor = (M_1_PI/8.0)/sqrt(gsl_pow_7(rho2 + sig2));
    F1 = factor*( 2.0*rho2*rho2 +  7.0*sig2*rho2 + 20.0*sig2*sig2);
    F2 = factor*(-6.0*rho2*rho2 - 27.0*sig2*rho2 - 21.0*sig2*sig2)/(rho2 + sig2);
};

inline void ENSTF(	const double &rho,
                     const double &sigma,
                     double &F1){
    //DOUT("----------------ENSTF()--------------");
    double rho2 = gsl_pow_2(rho);
    double sig2 = gsl_pow_2(sigma);
    F1 = (15.0*M_1_PI*sig2*sig2/8.0)/sqrt(gsl_pow_7(rho2 + sig2));
};

/*! \fn inline double ENST(const double &rho, const double &sigma, double &q)
 * \brief Compute enstrophy induced by vortex particle kernel
 * \param	sigma		double radius
 */
inline double ENST(	const double &sigma){
    //DOUT("----------------ENST()--------------");
	return 5.0*M_1_PI/2.0/gsl_pow_3(sigma);
};

inline double ENSTF(	const double &sigma){
    //DOUT("----------------ENST()--------------");
    return 15.0*M_1_PI/8.0/gsl_pow_3(sigma);
};

#endif
