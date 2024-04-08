/*! PArticle Wake ANalysis
 * \file wake_utils.h
 * \brief Inline functions for wake interaction calculations
 *
 * @author 	Puneet Singh
 * @date	04/15/2021
 *
 */

#ifndef WAKE_UTILS_H_
#define WAKE_UTILS_H_

#define HIGHORDER 0
#define GAUSSIAN 1

#include "src/utils/gsl_utils.h"
#include <gsl/gsl_blas.h>
#include <gsl/gsl_vector.h>
#include <chrono>
#include <thread>

#if HIGHORDER
#include "src/interaction/highorder.h"
#elif GAUSSIAN
#include "src/interaction/gaussian.h"
#endif

/*
 *
 * WAKE INTERACTION OPERATIONS
 *
 */

/*! \fn inline void VORTICITY(const double &kernel, const gsl_vector *strength, gsl_vector *vorticity )
 * \brief Compute vorticity induced by vortex particle kernel
 * \param	kernel		double
 * \param	strength	gsl vector particle vorticity strength
 * \param	vorticity	gsl vector output vorticity
 */
inline void VORTICITY(	const double &kernel, 
			const gsl_vector *strength, 
			gsl_vector *vorticity ){
	//DOUT("---------------VORTICITY()---------------");
	gsl_vector_memcpy(vorticity,strength);
	gsl_blas_dscal(kernel,vorticity);
};

/*! \fn inline void VELOCITY(const double &kernel, const gsl_vector *vorticity, const gsl_vector *displacement, gsl_vector *velocity )
 * \brief Compute velocity induced by vortex particle kernel
 * \param	kernel		double
 * \param	vorticity	gsl vector particle vorticity strength
 * \param	displacement	gsl vector displacement between particle and point
 * \param	velocity	gsl vector output velocity
 */
inline void VELOCITY(	const double &kernel, 
			const gsl_vector *vorticity, 
			const gsl_vector *displacement, 
			gsl_vector *velocity ){
	//DOUT("---------------VELOCITY()---------------");
	gsl_cross(vorticity,displacement,velocity);
	gsl_blas_dscal(kernel,velocity);
};

/*! \fn inline void VORSTRETCH(const double &q, const double &F, const gsl_vector *source_vorticity, const gsl_vector *target_vorticity, const gsl_vector *displacement, gsl_vector *retvorcity )
 * \brief Compute rate of change of vorticity due to vortex stretching
 * \param	q			q/rho kernel
 * \param	F			F kernel
 * \param	source_vorticity	gsl vector source vorticity
 * \param	target_vorticity	gsl vector target vorticity
 * \param	displacement		gsl vector displacement between source and target
 * \param	retvorcity		gsl vector output rate of change of vorticity
 */
inline void VORSTRETCH(	const double &q, 
			const double &F, 
			const gsl_vector *source_vorticity, 
			const gsl_vector *target_vorticity, 
			const gsl_vector *displacement, 
			gsl_vector *retvorcity ){
	//DOUT("---------------VORSTRETCH()---------------");
	// a_target x a_source
	gsl_vector *trgXsrc = gsl_vector_calloc(3);
	gsl_cross(target_vorticity,source_vorticity,trgXsrc);
	
	gsl_vector *crossed = gsl_vector_calloc(3);
	gsl_vector *stretch = gsl_vector_calloc(3);
	
	// da/dt = q*(a_trg x a_src)
	gsl_vector_memcpy(crossed,trgXsrc);
	gsl_blas_dscal(q,crossed);

	// da/dt = F*[disp.(a_trg x a_src)]disp
	double roaxa = 0.0;
	gsl_blas_ddot(displacement,trgXsrc,&roaxa);
	gsl_vector_memcpy(stretch,displacement);
	gsl_blas_dscal(F*roaxa,stretch);

	//gsl_vector_set_zero(retvorcity);
	gsl_vector_add(retvorcity,crossed);	
	gsl_vector_add(retvorcity,stretch);	
	
	gsl_vector_free(trgXsrc);
	gsl_vector_free(crossed);
	gsl_vector_free(stretch);

};

/*! \fn inline double ENSTROPHY(const double &s1, const double &s2, const gsl_vector *x1, const gsl_vector *x2, const gsl_vector *a1, const gsl_vector *a2)
 * \brief Compute helicity of vortices
 * \param	s	double source radius 
 * \param	a	gsl vector source vorticity
 */
inline double ENSTROPHY(const double &s,
			const gsl_vector *a){
	//DOUT("---------------ENSTROPHY()---------------");
	// Kernel Computation
	double a2 = 0.0;
	gsl_blas_ddot(a,a,&a2);
	double S = ENST(s)*a2;
	return S;
};

inline double ENSTROPHYF(const double &s,
                        const gsl_vector *a){
    //DOUT("---------------ENSTROPHY()---------------");
    // Kernel Computation
    double a2 = 0.0;
    gsl_blas_ddot(a,a,&a2);
    double S = ENSTF(s)*a2;
    return S;
};

/*! \fn inline double ENSTROPHY(const double &s1, const double &s2, const gsl_vector *x1, const gsl_vector *x2, const gsl_vector *a1, const gsl_vector *a2)
 * \brief Compute helicity of vortices
 * \param	s1	double source radius 
 * \param	s2	double target radius 
 * \param	x1	gsl vector source position 
 * \param	x2	gsl vector target position 
 * \param	a1	gsl vector source vorticity
 * \param	a2	gsl vector target vorticity
 */
inline double ENSTROPHY(const double &s1,
			const double &s2,
			const gsl_vector *x1, 
			const gsl_vector *x2, 
			const gsl_vector *a1, 
			const gsl_vector *a2){
	//DOUT("---------------ENSTROPHY()---------------");
	//std::this_thread::sleep_for(std::chrono::milliseconds(2000));
	// Kernel Computation
	gsl_vector *x12 = gsl_vector_calloc(3);
	gsl_vector_memcpy(x12,x2);
	gsl_vector_sub(x12,x1);
	double rho = gsl_blas_dnrm2(x12);
	double sigma = sqrt(0.5*(gsl_pow_2(s1) + gsl_pow_2(s2)));
	
	double F1 = 0.0;
	double F2 = 0.0;

	ENST(rho,sigma,F1,F2);

	// (a1.a2)
	double a1a2 = 0.0;;
	gsl_blas_ddot(a1,a2,&a1a2);
	// (a1.x12)
	double a1x12 = 0.0;;
	gsl_blas_ddot(a1,x12,&a1x12);
	// (a2.x12)
	double a2x12 = 0.0;;
	gsl_blas_ddot(a2,x12,&a2x12);
	// (a1.x12).(a2.x12)
	
	double S = F1*a1a2 + F2*a1x12*a2x12;
    gsl_vector_free(x12);

	return S;
};

inline double ENSTROPHYF(const double &s1,
                        const double &s2,
                        const gsl_vector *x1,
                        const gsl_vector *x2,
                        const gsl_vector *a1,
                        const gsl_vector *a2){
    // Kernel Computation
    gsl_vector *x12 = gsl_vector_calloc(3);
    gsl_vector_memcpy(x12,x2);
    gsl_vector_sub(x12,x1);
    double rho = gsl_blas_dnrm2(x12);
    double sigma = sqrt(0.5*(gsl_pow_2(s1) + gsl_pow_2(s2)));

    double F1 = 0.0;

    ENSTF(rho,sigma,F1);

    // (a1.a2)
    double a1a2 = 0.0;;
    gsl_blas_ddot(a1,a2,&a1a2);

    double S = F1*a1a2;
    gsl_vector_free(x12);

    return S;
};

/*! \fn inline double HELICITY(const double &s1, const double &s2, const gsl_vector *x1, const gsl_vector *x2, const gsl_vector *a1, const gsl_vector *a2)
 * \brief Compute helicity of vortices
 * \param	s1	double source radius 
 * \param	s2	double target radius 
 * \param	x1	gsl vector source position 
 * \param	x2	gsl vector target position 
 * \param	a1	gsl vector source vorticity
 * \param	a2	gsl vector target vorticity
 */
inline double HELICITY(	const double &s1,
			const double &s2,
			const gsl_vector *x1, 
			const gsl_vector *x2, 
			const gsl_vector *a1, 
			const gsl_vector *a2){
	//DOUT("---------------HELICITY()---------------");
	//std::this_thread::sleep_for(std::chrono::milliseconds(2000));
	// Kernel Computation
	gsl_vector *x12 = gsl_vector_calloc(3);
	gsl_vector_memcpy(x12,x2);
	gsl_vector_sub(x12,x1);
	double rho = gsl_blas_dnrm2(x12);
	double sigma = sqrt(0.5*(gsl_pow_2(s1) + gsl_pow_2(s2)));
	double q = QSIG(rho,sigma);
	// a1 x a2
	gsl_vector *a1Xa2 = gsl_vector_calloc(3);
	gsl_cross(a1,a2,a1Xa2);
	
	// x12.(a1 x a2)
	double roaxa = 0.0;;
	gsl_blas_ddot(x12,a1Xa2,&roaxa);

	double H = q*roaxa;
    gsl_vector_free(x12);
    gsl_vector_free(a1Xa2);

    return H;
};

/*! \fn inline double KINETICENERGY(const double &s, const gsl_vector *x, const gsl_vector *a)
 * \brief Compute kinetic energy of vortices
 * \param	s	double source radius
 * \param	a	gsl vector source vorticity
 */
inline double KINETICENERGY(	const double &s,
				const gsl_vector *a){
	//DOUT("---------------KINETICENERGY()---------------");
	double a2 = 0.0;
	gsl_blas_ddot(a,a,&a2);
	// KE = (1/8 pi).(a1.a1/sigma)
	double KE = M_1_PI*a2/s/8.0;
	return KE;
};

inline double KINETICENERGYF(	const double &s,
                                const gsl_vector *a){
    //DOUT("---------------KINETICENERGYF()---------------");
    double a2 = 0.0;
    gsl_blas_ddot(a,a,&a2);
    // KE = (3/16 pi).(a1.a1/sigma)
    double KE = 3*M_1_PI*a2/(s*16.0);
    return KE;
};

/*! \fn inline double KINETICENERGY(const double &s1, const double &s2, const gsl_vector *x1, const gsl_vector *x2, const gsl_vector *a1, const gsl_vector *a2)
 * \brief Compute kinetic energy of vortices
 * \param	s1	double source radius 
 * \param	s2	double target radius 
 * \param	x1	gsl vector source position 
 * \param	x2	gsl vector target position 
 * \param	a1	gsl vector source vorticity
 * \param	a2	gsl vector target vorticity
 */
inline double KINETICENERGY(	const double &s1,
				const double &s2,
				const gsl_vector *x1, 
				const gsl_vector *x2, 
				const gsl_vector *a1, 
				const gsl_vector *a2){
	//DOUT("---------------KINETICENERGY()---------------");
	// Kernel Computation
	gsl_vector *x12 = gsl_vector_calloc(3);
	gsl_vector_memcpy(x12,x2);
	gsl_vector_sub(x12,x1);
	double rho2;
	gsl_blas_ddot(x12,x12,&rho2);
	double sigma2 = 0.5*(gsl_pow_2(s1) + gsl_pow_2(s2));
	// a1 . a2
	double a1a2 = 0.0;		
	gsl_blas_ddot(a1,a2,&a1a2);
	
	double x12a1 = 0.0;
	gsl_blas_ddot(x12,a1,&x12a1);
	double x12a2 = 0.0;
	gsl_blas_ddot(x12,a2,&x12a2);
	
	// KE = (1/16 pi).((rho^2 + 2.sigma^2)*(a1.a2) + (x12.a1)(x12.a2))/(rho^2 + sigma^2)^3/2
	double KE = ((rho2 + 2.0*sigma2)*a1a2 + (x12a1*x12a2))/pow(rho2 + sigma2,1.5)/16.0/M_PI;

    gsl_vector_free(x12);
	return KE;
};

inline double KINETICENERGYF(	const double &s1,
                                const double &s2,
                                const gsl_vector *x1,
                                const gsl_vector *x2,
                                const gsl_vector *a1,
                                const gsl_vector *a2){
    //DOUT("---------------KINETICENERGYF()---------------");
    // Kernel Computation
    gsl_vector *x12 = gsl_vector_calloc(3);
    gsl_vector_memcpy(x12,x2);
    gsl_vector_sub(x12,x1);
    double rho2;
    gsl_blas_ddot(x12,x12,&rho2);
    double sigma2 = 0.5*(gsl_pow_2(s1) + gsl_pow_2(s2));
    // a1 . a2
    double a1a2 = 0.0;
    gsl_blas_ddot(a1,a2,&a1a2);

    // KE = (1/8 pi).(rho^2 + 1.5*sigma^2)*(a1.a2) /(rho^2 + sigma^2)^3/2
    double KE = (rho2 + 1.5*sigma2)*a1a2 /pow(rho2 + sigma2,1.5)/8.0/M_PI;
    gsl_vector_free(x12);

    return KE;
};

/*! \fn inline void DIFFUSION(	const double &nu, const double &sigma, const double &Z, const gsl_vector *source_vorticity, const gsl_vector *target_vorticity, const double &source_volume, const double &target_volume, gsl_vector *retvorcity )
 * \brief Compute rate of change of vorticity due to viscous diffusion
 * \param	nu			double viscosity
 * \param	sigma			double smoothing radius
 * \param	n			double eta kernel
 * \param	source_vorticity	gsl vector source vorticity
 * \param	target_vorticity	gsl vector target vorticity
 * \param	source_volume		double source volume
 * \param	target_volume		double target volume
 * \param	retvorcity		gsl vector output rate of change of vorticity
 */
inline void DIFFUSION(	const double &nu,
			const double &sigma, 
			const double &n,
			const gsl_vector *source_vorticity, 
			const gsl_vector *target_vorticity, 
			const double &source_volume, 
			const double &target_volume,
			gsl_vector *retvorcity ){

	//DOUT("---------------DIFFUSION()---------------");
	gsl_vector *va12 = gsl_vector_calloc(3);
	gsl_vector *va21 = gsl_vector_calloc(3);
	gsl_vector *dva = gsl_vector_calloc(3);
	
	// va12 = volume_target*vorticity_source
	gsl_vector_memcpy(va12,source_vorticity);
	gsl_blas_dscal(target_volume,va12);
	
	// va21 = volume_source*vorticity_target
	gsl_vector_memcpy(va21,target_vorticity);
	gsl_blas_dscal(source_volume,va21);
	
	// dva = 2*nu*Z*(va12 - va21)/sigma^2
	double sig12 = 0.5*sigma*sigma;
	gsl_vector_memcpy(dva,va12);
	gsl_vector_sub(dva,va21);
	gsl_blas_dscal(n*nu/sig12,dva);
	
	// da = da + dva
	gsl_vector_add(retvorcity,dva);
	
	gsl_vector_free(va12);
	gsl_vector_free(va21);
	gsl_vector_free(dva);
};

/*! \fn inline void INTERACT(const double &nu, const double &sigma, const gsl_vector *r_source, const gsl_vector *r_target, const gsl_vector *a_source, const gsl_vector *a_target, const double &v_source, const double &v_target, gsl_vector *dr_source, gsl_vector *dr_target, gsl_vector *da_source, gsl_vector *da_target)
 * \brief Compute interaction between two vortex particles
 * \param	nu			double viscosity
 * \param	sigma			double smoothing radius
 * \param	r_source		gsl vector source position
 * \param	r_target		gsl vector target position
 * \param	a_source		gsl vector source vorticity
 * \param	a_target		gsl vector target vorticity
 * \param	v_source		double source volume
 * \param	v_target		double target volume
 * \param	dr_source		gsl vector source velocity 
 * \param	dr_target		gsl vector target velocity 
 * \param	da_source		gsl vector source rate of change of vorticity
 * \param	da_target		gsl vector target rate of change of vorticity
 */
inline void INTERACT(	const double &nu,
			const double &sigma,
			const gsl_vector *r_source, 
			const gsl_vector *r_target, 
			const gsl_vector *a_source, 
			const gsl_vector *a_target, 
			const double &v_source, 
			const double &v_target, 
			gsl_vector *dr_source, 
			gsl_vector *dr_target,
			gsl_vector *da_source, 
			gsl_vector *da_target){
	//DOUT("---------------INTERACT()---------------");
	// Kernel Computation
	gsl_vector *displacement = gsl_vector_calloc(3);
	gsl_vector_memcpy(displacement,r_target);
	gsl_vector_sub(displacement,r_source);
	double rho = gsl_blas_dnrm2(displacement);
	double q = 0.0, F = 0.0, Z = 0.0, n = 0.0;
	KERNEL(rho,sigma,q,F,Z,n);

	// Velocity computation
	gsl_vector *dr = gsl_vector_calloc(3);
	// Target
	VELOCITY(q,a_source,displacement,dr);
	gsl_vector_add(dr_target,dr);
	// Source
	VELOCITY(-q,a_target,displacement,dr);
	gsl_vector_add(dr_source,dr);

	// Rate of change of vorticity computation
	gsl_vector *da = gsl_vector_calloc(3);
	VORSTRETCH(q,F,a_source,a_target,displacement,da);
	DIFFUSION(nu,sigma,n,a_source,a_target,v_source,v_target,da);
	// Target
	gsl_vector_add(da_target,da);
	// Source
	gsl_vector_sub(da_source,da);
	
	// Clean up
	gsl_vector_free(dr);
	gsl_vector_free(da);
	gsl_vector_free(displacement);
};

/*! \fn inline void INTERACT(const double &nu,const double &s_source,const double &s_target,const gsl_vector *r_source,const gsl_vector *r_target,const gsl_vector *a_source,const gsl_vector *a_target, const double &v_source, const double &v_target, gsl_vector *dr_source, gsl_vector *dr_target,gsl_vector *da_source, gsl_vector *da_target)
 * \brief Compute interaction between two vortex particles
 * \param	nu			double viscosity
 * \param	s_source		double smoothing radius of source
 * \param	s_target		double smoothing radius of target
 * \param	r_source		gsl vector source position
 * \param	r_target		gsl vector target position
 * \param	a_source		gsl vector source vorticity
 * \param	a_target		gsl vector target vorticity
 * \param	v_source		double source volume
 * \param	v_target		double target volume
 * \param	dr_source		gsl vector source velocity 
 * \param	dr_target		gsl vector target velocity 
 * \param	da_source		gsl vector source rate of change of vorticity
 * \param	da_target		gsl vector target rate of change of vorticity
 */
inline void INTERACT(	const double &nu,
			const double &s_source,
			const double &s_target,
			const gsl_vector *r_source, 
			const gsl_vector *r_target, 
			const gsl_vector *a_source, 
			const gsl_vector *a_target, 
			const double &v_source, 
			const double &v_target, 
			gsl_vector *dr_source, 
			gsl_vector *dr_target,
			gsl_vector *da_source, 
			gsl_vector *da_target){
	//DOUT("---------------INTERACT()---------------");
	// Kernel Computation
	gsl_vector *displacement = gsl_vector_calloc(3);
	gsl_vector_memcpy(displacement,r_target);
	gsl_vector_sub(displacement,r_source);
	double rho = gsl_blas_dnrm2(displacement);
	double q = 0.0, F = 0.0, Z = 0.0, n = 0.0;
	double sigma = sqrt(0.5*(gsl_pow_2(s_source) + gsl_pow_2(s_target)));

	// Velocity computation
	gsl_vector *dr = gsl_vector_calloc(3);
	// Target
	KERNEL(rho,sigma,q,F,Z,n);
	VELOCITY(q,a_source,displacement,dr);
	gsl_vector_add(dr_target,dr);
	// Source
	VELOCITY(-q,a_target,displacement,dr);
	gsl_vector_add(dr_source,dr);

	// Rate of change of vorticity computation
	gsl_vector *da = gsl_vector_calloc(3);
	VORSTRETCH(q,F,a_source,a_target,displacement,da);
	DIFFUSION(nu,sigma,n,a_source,a_target,v_source,v_target,da);
	// Target
	gsl_vector_add(da_target,da);
	// Source
	gsl_vector_sub(da_source,da);
	
	// Clean up 
	gsl_vector_free(dr);
	gsl_vector_free(da);
	gsl_vector_free(displacement);
};

/*! \fn inline void INTERACT(const double &nu,const double &s_source,const double &s_target,const gsl_vector *r_source,const gsl_vector *r_target,const gsl_vector *a_source,const gsl_vector *a_target, const double &v_source, const double &v_target, gsl_vector *dr_source, gsl_vector *da_source, r *da_target)
 * \brief Compute interaction between two vortex particles
 * \param	nu			double viscosity
 * \param	s_source		double smoothing radius of source
 * \param	s_target		double smoothing radius of target
 * \param	r_source		gsl vector source position
 * \param	r_target		gsl vector target position
 * \param	a_source		gsl vector source vorticity
 * \param	a_target		gsl vector target vorticity
 * \param	v_source		double source volume
 * \param	v_target		double target volume
 * \param	dr_target		gsl vector target velocity 
 * \param	da_target		gsl vector target rate of change of vorticity
 * \param	vx_source		double source x velocity 
 * \param	vy_source		double source y velocity 
 * \param	vz_source		double source z velocity 
 * \param	qx_source		double source x vorticity 
 * \param	qy_source		double source y vorticity 
 * \param	qz_source		double source z vorticity 
 */
inline void INTERACT(	const double &nu,
			const double &s_source,
			const double &s_target,
			const gsl_vector *r_source, 
			const gsl_vector *r_target, 
			const gsl_vector *a_source, 
			const gsl_vector *a_target, 
			const double &v_source, 
			const double &v_target, 
			gsl_vector *dr_target, 
			gsl_vector *da_target, 
			double &vx_source,
			double &vy_source,
			double &vz_source,
			double &qx_source,
			double &qy_source,
			double &qz_source){
	//DOUT("---------------INTERACT()---------------");
	// Kernel Computation
	gsl_vector *displacement = gsl_vector_calloc(3);
	gsl_vector_memcpy(displacement,r_target);
	gsl_vector_sub(displacement,r_source);
	double rho = gsl_blas_dnrm2(displacement);
	double q = 0.0, F = 0.0, Z = 0.0, n = 0.0;
	double sigma = sqrt(0.5*(gsl_pow_2(s_source) + gsl_pow_2(s_target)));

	// Velocity computation
	gsl_vector *dr = gsl_vector_calloc(3);
	// Target
	KERNEL(rho,sigma,q,F,Z,n);
	VELOCITY(q,a_source,displacement,dr);
	gsl_vector_add(dr_target,dr);
	// Source
	VELOCITY(-q,a_target,displacement,dr);
	vx_source = gsl_vector_get(dr,0);
	vy_source = gsl_vector_get(dr,1);
	vz_source = gsl_vector_get(dr,2);

	// Rate of change of vorticity computation
	gsl_vector *da = gsl_vector_calloc(3);
	VORSTRETCH(q,F,a_source,a_target,displacement,da);
	DIFFUSION(nu,sigma,n,a_source,a_target,v_source,v_target,da);
	// Target
	gsl_vector_add(da_target,da);
	// Source
	qx_source = -gsl_vector_get(da,0);
	qy_source = -gsl_vector_get(da,1);
	qz_source = -gsl_vector_get(da,2);
	
	// Clean up
	gsl_vector_free(dr);
	gsl_vector_free(da);
	gsl_vector_free(displacement);
};

/*! \fn inline void INFLUENCE(const double &sigma, const gsl_vector *r_source, const gsl_vector *r_target, const gsl_vector *a_source, const gsl_vector *a_target, gsl_vector *k_source, gsl_vector *k_target)
 * \brief Compute interaction between two vortex particles
 * \param	sigma			double smoothing radius
 * \param	r_source		gsl vector source position
 * \param	r_target		gsl vector target position
 * \param	a_source		gsl vector source vorticity strength
 * \param	a_target		gsl vector target vorticity strength
 * \param	k_source		gsl vector source vorticity field
 * \param	k_target		gsl vector target vorticity field
 */
inline void INFLUENCE(	const double &sigma,
			const gsl_vector *r_source, 
			const gsl_vector *r_target, 
			const gsl_vector *a_source, 
			const gsl_vector *a_target, 
			gsl_vector *k_source, 
			gsl_vector *k_target){
	//DOUT("---------------INFLUENCE()---------------");
	// Kernel Computation
	gsl_vector *displacement = gsl_vector_calloc(3);
	gsl_vector_memcpy(displacement,r_target);
	gsl_vector_sub(displacement,r_source);
	double rho = gsl_blas_dnrm2(displacement);
	double Z = ZETASIG(rho,sigma);
	// Vorticity computation
	gsl_vector *dk = gsl_vector_calloc(3);
	VORTICITY(Z,a_source,dk);
	// Target
	gsl_vector_add(k_target,dk);
	// Source
	VORTICITY(Z,a_target,dk);
	gsl_vector_add(k_source,dk);
	// Clean up
	gsl_vector_free(dk);
	gsl_vector_free(displacement);
};

/*! \fn inline void INFLUENCE(const double &s_source, const double &s_target, const gsl_vector *r_source, const gsl_vector *r_target, const gsl_vector *a_source, const gsl_vector *a_target, gsl_vector *k_source, gsl_vector *k_target)
 * \brief Compute interaction between two vortex particles
 * \param	s_source		double smoothing radius of source
 * \param	s_target		double smoothing radius of target
 * \param	r_source		gsl vector source position
 * \param	r_target		gsl vector target position
 * \param	a_source		gsl vector source vorticity strength
 * \param	a_target		gsl vector target vorticity strength
 * \param	k_source		gsl vector source vorticity field
 * \param	k_target		gsl vector target vorticity field
 */
inline void INFLUENCE(	const double &s_source,
			const double &s_target,
			const gsl_vector *r_source, 
			const gsl_vector *r_target, 
			const gsl_vector *a_source, 
			const gsl_vector *a_target, 
			gsl_vector *k_source, 
			gsl_vector *k_target){
	//DOUT("---------------INFLUENCE()---------------");
	double sigma = sqrt(0.5*(gsl_pow_2(s_source) + gsl_pow_2(s_target)));
	// Kernel Computation
	gsl_vector *displacement = gsl_vector_calloc(3);
	gsl_vector_memcpy(displacement,r_target);
	gsl_vector_sub(displacement,r_source);
	double rho = gsl_blas_dnrm2(displacement);
	double Z = ZETASIG(rho,sigma);
	// Vorticity computation
	gsl_vector *dk = gsl_vector_calloc(3);
	VORTICITY(Z,a_source,dk);
	// Target
	gsl_vector_add(k_target,dk);
	// Source
	VORTICITY(Z,a_target,dk);
	gsl_vector_add(k_source,dk);
	// Clean up
	gsl_vector_free(dk);
	gsl_vector_free(displacement);
};

inline void INFLUENCE(	const double &s_source,
			const double &s_target,
			const gsl_vector *r_source, 
			const gsl_vector *r_target, 
			const gsl_vector *a_source, 
			const gsl_vector *a_target, 
			gsl_vector *k_target, 
			double &kx_source,
			double &ky_source,
			double &kz_source){
	//DOUT("---------------INFLUENCE()---------------");
	double sigma = sqrt(0.5*(gsl_pow_2(s_source) + gsl_pow_2(s_target)));
	// Kernel Computation
	gsl_vector *displacement = gsl_vector_calloc(3);
	gsl_vector_memcpy(displacement,r_target);
	gsl_vector_sub(displacement,r_source);
	double rho = gsl_blas_dnrm2(displacement);
	double Z = ZETASIG(rho,sigma);
	// Vorticity computation
	gsl_vector *dk = gsl_vector_calloc(3);
	VORTICITY(Z,a_source,dk);
	// Target
	gsl_vector_add(k_target,dk);
	// Source
	VORTICITY(Z,a_target,dk);
	// Source
	kx_source = gsl_vector_get(dk,0);
	ky_source = gsl_vector_get(dk,1);
	kz_source = gsl_vector_get(dk,2);
	// Clean up
	gsl_vector_free(dk);
	gsl_vector_free(displacement);
};

inline void SELFINFLUENCE(	const double &s_target,
				const gsl_vector *r_target, 
				const gsl_vector *a_target, 
				gsl_vector *k_target){
	//DOUT("---------------SELFINFLUENCE()---------------");
	// Kernel Computation
	double Z = ZETASIG(0.0,s_target);
	// Vorticity computation
	gsl_vector *dk = gsl_vector_calloc(3);
	VORTICITY(Z,a_target,dk);
	// Target
	gsl_vector_add(k_target,dk);
	// Clean up
	gsl_vector_free(dk);
};

#endif
