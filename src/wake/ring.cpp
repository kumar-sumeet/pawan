/*! PArticle Wake ANalysis
 * \file ring.cpp
 * \brief Routines for creating Vortex rings
 *
 * @author Puneet Singh
 * @date 04/24/2021
 */
#include "ring.h"

pawan::__ring::__ring(const double& gamma, const double& radius, const double& core, const int& nRadial) {
    create_particles(nRadial);
    double dPsi = 2.0 * M_PI / nRadial;                 /*!< Azimuthal step size */
    double strength = gamma * radius * dPsi;            /*!< Vorticity magnitude */
    double volume = 4.0 * M_PI * gsl_pow_3(core) / 3.0; /*!< Volume of annular segment */
    for (int i = 0; i < _numParticles; ++i) {
        double psi = i * dPsi;
        gsl_matrix_set(_position, i, 0, radius * cos(psi));
        gsl_matrix_set(_position, i, 1, radius * sin(psi));
        gsl_matrix_set(_position, i, 2, 0.0);
        gsl_matrix_set(_vorticity, i, 0, -strength * sin(psi));
        gsl_matrix_set(_vorticity, i, 1, strength * cos(psi));
        gsl_matrix_set(_vorticity, i, 2, 0.0);
        gsl_vector_set(_radius, i, core);
        gsl_vector_set(_volume, i, volume);
        gsl_vector_set(_birthstrength, i, strength);
    }
}

// pawan::__ring::__ring(const double &gamma, const double &radius, const double &core, const int &nRadial, const int &nCore){
// create_particles(nRadial*nCore);
// double dPsi = 2.0*M_PI/nRadial; 		[>!< Azimuthal step size <]
// double K = -log(0.01);
// double Phi = 0.5*(1.0+sqrt(5.0));
// double strength = gamma*core*core/K;		[>!< Vorticity magnitude <]
// double sigma = pow(2*gsl_pow_2(M_PI*core)*radius/_numParticles,1./3.);[>!< Smoothing radius <]
// double volume = 4.0*M_PI*gsl_pow_3(sigma)/3.0;	[>!< Volume of annular segment <]
// double mag0 = strength*(1-exp(-K/nCore))*nCore/gsl_pow_2(core);
// for(int i = 0; i<nRadial; ++i){
// double psi = i*dPsi;
// double sPsi = sin(psi);
// double cPsi = cos(psi);

// gsl_matrix_set(_position,i*nCore,0,radius*cPsi);
// gsl_matrix_set(_position,i*nCore,1,radius*sPsi);
// gsl_matrix_set(_position,i*nCore,2,0.0);
// gsl_matrix_set(_vorticity,i*nCore,0,-mag0*sPsi);
// gsl_matrix_set(_vorticity,i*nCore,1,mag0*cPsi);
// gsl_matrix_set(_vorticity,i*nCore,2,0.0);
// gsl_vector_set(_radius,i*nCore,sigma);
// gsl_vector_set(_volume,i*nCore,volume);
// gsl_vector_set(_birthstrength,i*nCore,mag0);

// for(int j = 1; j<nCore; ++j){
// double r = core*sqrt((double)j/nCore);
// double theta = 2.0*M_PI*j/gsl_pow_2(Phi);
// double x = (radius + r*cos(theta))*cPsi;
// double y = (radius + r*cos(theta))*sPsi;
// double z = r*sin(theta);
// double dr = 0.5*radius/sqrt((j+1)*nCore);
// double mag = strength*exp(-K*gsl_pow_2(r/core))*sinh(K*r*dr/gsl_pow_2(core))/core/dr;
// double ax = -mag*sPsi;
// double ay = mag*cPsi;
// size_t index = i*nCore + j;
// gsl_matrix_set(_position,index,0,x);
// gsl_matrix_set(_position,index,1,y);
// gsl_matrix_set(_position,index,2,z);
// gsl_matrix_set(_vorticity,index,0,ax);
// gsl_matrix_set(_vorticity,index,1,ay);
// gsl_matrix_set(_vorticity,index,2,0.0);
// gsl_vector_set(_radius,index,sigma);
// gsl_vector_set(_volume,index,volume);
// gsl_vector_set(_birthstrength,index,mag);
//}
//}
//}

pawan::__ring::__ring(const double& gamma, const double& radius, const double& core, const int& nRadial, const int& nLayers, const bool& shift) {
    create_particles(nRadial * gsl_pow_2(2 * nLayers - 1));
    double dPsi = 2.0 * M_PI / nRadial; /*!< Azimuthal step size */
    double K = -log(0.01);
    double core2 = gsl_pow_2(core);
    double strength = gamma * core2 / K;                                              /*!< Vorticity magnitude */
    double r0 = core / (2 * nLayers - 1);                                             /*!< Central area */
    double sigma = pow(2 * gsl_pow_2(M_PI * core) * radius / _numParticles, 1. / 3.); /*!< Smoothing radius */
    double volume = 4.0 * M_PI * gsl_pow_3(sigma) / 3.0;                              /*!< Volume of annular segment */
    size_t index = 0;
    double mag0 = strength * (1.0 - exp(-K * gsl_pow_2(r0 / core))) / (r0 * r0);
    double delpsi = shift ? 0.5 * dPsi : 0.;
    for (size_t i = 0; i < nRadial; ++i) {
        double psi = i * dPsi + delpsi;
        double sPsi = sin(psi);
        double cPsi = cos(psi);

        gsl_matrix_set(_position, index, 0, radius * cPsi);
        gsl_matrix_set(_position, index, 1, radius * sPsi);
        gsl_matrix_set(_position, index, 2, 0.0);
        gsl_matrix_set(_vorticity, index, 0, -mag0 * sPsi);
        gsl_matrix_set(_vorticity, index, 1, mag0 * cPsi);
        gsl_matrix_set(_vorticity, index, 2, 0.0);
        gsl_vector_set(_radius, index, sigma);
        gsl_vector_set(_volume, index, volume);
        gsl_vector_set(_birthstrength, index, mag0);
        index++;
        for (size_t j = 1; j < nLayers; ++j) {
            double r = r0 * (12. * gsl_pow_2(j) + 1.0) / (6. * j);
            double dr = r0 * (2. - 1. / (6. * j * (j + 1.)));
            size_t nCore = 8 * j;
            double dTheta = 2. * M_PI / nCore;
            for (size_t k = 0; k < nCore; ++k) {
                double theta = k * dTheta;
                double x = (radius + r * cos(theta)) * cPsi;
                double y = (radius + r * cos(theta)) * sPsi;
                double z = r * sin(theta);
                double mag = strength * exp(-K * gsl_pow_2(r / core)) * sinh(K * r * dr / core2) / core / dr;
                double ax = -mag * sPsi;
                double ay = mag * cPsi;
                gsl_matrix_set(_position, index, 0, x);
                gsl_matrix_set(_position, index, 1, y);
                gsl_matrix_set(_position, index, 2, z);
                gsl_matrix_set(_vorticity, index, 0, ax);
                gsl_matrix_set(_vorticity, index, 1, ay);
                gsl_matrix_set(_vorticity, index, 2, 0.0);
                gsl_vector_set(_radius, index, sigma);
                gsl_vector_set(_volume, index, volume);
                gsl_vector_set(_birthstrength, index, mag);
                index++;
            }
        }
    }
}

pawan::__ring::~__ring() {
}
