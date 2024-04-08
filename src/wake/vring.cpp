/*! PArticle Wake ANalysis
 * \file vring.cpp
 * \brief Routines for creating Vortex rings
 *
 * @author Puneet Singh
 * @date 08/12/2021
 */
#include "vring.h"

void pawan::__vring::create_particles(const int &n){
	_numParticles = n;
    _maxnumParticles = _numParticles;
    _maxsize         = _size; //not a clean fix
	_position = gsl_matrix_calloc(_numParticles,3);
	_velocity = gsl_matrix_calloc(_numParticles,3);
	_vorticity = gsl_matrix_calloc(_numParticles,3);
	_retvorcity = gsl_matrix_calloc(_numParticles,3);
	_radius = gsl_vector_calloc(_numParticles);
	_volume = gsl_vector_calloc(_numParticles);
	_birthstrength = gsl_vector_calloc(_numParticles);
	_vorticityfield = gsl_matrix_calloc(_numParticles,3);
    _active = gsl_vector_calloc(_numParticles);//added for compliance with single plot script
}

void pawan::__vring::set_vorticity(){
	size_t indexP = 0;	
	for(size_t i = 0; i<_nAzi; ++i){
		size_t indexV = 0;	
		double psi = i*_dPsi;
		double sPsi = sin(psi);
		double cPsi = cos(psi);
		// Central particle
		gsl_matrix_set(_vorticity,indexP,0,-sPsi*gsl_vector_get(_strength,indexV));
		gsl_matrix_set(_vorticity,indexP,1,cPsi*gsl_vector_get(_strength,indexV));
		gsl_matrix_set(_vorticity,indexP,2,0.0);
		indexP++;
		indexV++;
		for(size_t j = 1; j<=_nLayers; ++j){
			size_t nCore = 8*j;
			for(size_t k = 0; k<nCore; ++k){
				double ax = -sPsi*gsl_vector_get(_strength,indexV);
				double ay = cPsi*gsl_vector_get(_strength,indexV);
				// Particles in layer
				gsl_matrix_set(_vorticity,indexP,0,ax);
				gsl_matrix_set(_vorticity,indexP,1,ay);
				gsl_matrix_set(_vorticity,indexP,2,0.0);
				indexP++;
				indexV++;
			}
		}
	}
}

pawan::__vring::__vring(const double &radius,
                        const double &core,
                        const int &nLayers,
                        const int &nAzi,
                        const double &sig){
	_R = radius;
	_sigmaR = core;
	_nLayers = nLayers;
	_size = gsl_pow_2(2*_nLayers + 1);
    OUT("_size",_size);
	_strength = gsl_vector_calloc(_size);
    //double r0 = _sigmaR/(2*_nLayers + 1);		/*!< Central area */
    double r0 = 0.35/(2*_nLayers + 1);
    OUT("r0",r0);
	//_nAzi = ceil(M_PI*_R/r0);
	_nAzi = nAzi;
	create_particles(_nAzi*_size);
    OUT("Number of particles",_nAzi*_size);

	_dPsi = 2.0*M_PI/_nAzi;				/*!< Azimuthal step size */ 
	//double sigma = 1.3*(2.0*r0);//changed this from 1.5 to 1.3	/*!< Smoothing radius */
    double sigma = sig;
	OUT("sigma",sigma);
	double volume = 0.0;	
	size_t index = 0;	
	for(size_t i = 0; i<_nAzi; ++i){
		double psi = i*_dPsi;
		double sPsi = sin(psi);
		double cPsi = cos(psi);

		volume = M_PI*gsl_pow_2(r0)*_R*_dPsi;
        //center vortex particle
		gsl_matrix_set(_position,index,0,_R*cPsi);
		gsl_matrix_set(_position,index,1,_R*sPsi);
		gsl_matrix_set(_position,index,2,0.0);
		gsl_vector_set(_radius,index,sigma);
		gsl_vector_set(_volume,index,volume);
		gsl_vector_set(_birthstrength,index,1.0);
		index++;
		//vortex particles in subsequent layers
		for(size_t j = 1; j<=_nLayers; ++j){
			double r = r0*(12.*gsl_pow_2(j)+1.0)/(6.*j); //centroid location of cell
			size_t nCore = 8*j;                          //nth layer has 8n particles
			double dTheta = 2.*M_PI/nCore;
			for(size_t k = 0; k<nCore; ++k){
				double theta = k*dTheta;
				double x = (_R + r*cos(theta))*cPsi;
				double y = (_R + r*cos(theta))*sPsi;
				double z = r*sin(theta);
				double ax = -sPsi;
				double ay = cPsi;
				volume = 2.0*r0*_dPsi*(2.0*dTheta*j*_R*r0 + (12.0*gsl_pow_2(j)+1.0)*gsl_pow_2(r0)*(sin(theta + dTheta) - sin(theta))/3.0);
				gsl_matrix_set(_position,index,0,x);
				gsl_matrix_set(_position,index,1,y);
				gsl_matrix_set(_position,index,2,z);
				gsl_vector_set(_radius,index,sigma);
				gsl_vector_set(_volume,index,volume);
				gsl_vector_set(_birthstrength,index,1.0);
				index++;
			}
		}
	}
	set_vorticity();
}

/*
pawan::__vring::__vring(const double &radius,
                        const double &core,
                        const int &nLayers){
	_R = radius;
	_sigmaR = core;
	_nLayers = nLayers;
	_size = gsl_pow_2(2*_nLayers + 1);
	_strength = gsl_vector_calloc(_size);
    double r0 = _sigmaR/(2*_nLayers + 1);		// Central area
	_nAzi = ceil(M_PI*_R/r0);
	_dPsi = 2.0*M_PI/_nAzi;				//Azimuthal step size

    initialise_memory();
    addParticles();
}

pawan::__vring::__vring(const double &radius,
                        const double &core,
                        const int &nLayers,
                        const int &nAzi){
    _R = radius;
    _sigmaR = core;
    _nLayers = nLayers;
    _size = gsl_pow_2(2*_nLayers + 1);
    _strength = gsl_vector_calloc(_size);
    _nAzi = nAzi;
    _dPsi = 2.0*M_PI/_nAzi;				// Azimuthal step size

    initialise_memory();
    addParticles();
}

void pawan::__vring::addParticles(){
    double r0 = _sigmaR/(2*_nLayers + 1);		// Central area
    double sigma = 1.5*(2.0*r0);			// Smoothing radius
    OUT("sigma",sigma);
    double volume = 0.0;
    size_t index = 0;
    for(size_t i = 0; i<_nAzi; ++i){
        double psi = i*_dPsi;
        double sPsi = sin(psi);
        double cPsi = cos(psi);

        volume = M_PI*gsl_pow_2(r0)*_R*_dPsi;
        //center vortex particle
        gsl_matrix_set(_position,index,0,_R*cPsi);
        gsl_matrix_set(_position,index,1,_R*sPsi);
        gsl_matrix_set(_position,index,2,0.0);
        gsl_vector_set(_radius,index,sigma);
        gsl_vector_set(_volume,index,volume);
        gsl_vector_set(_birthstrength,index,1.0);
        index++;
        //vortex particles in subsequent layers
        for(size_t j = 1; j<=_nLayers; ++j){
            double r = r0*(12.*gsl_pow_2(j)+1.0)/(6.*j); //centroid location of cell
            size_t nCore = 8*j;                          //nth layer has 8n particles
            double dTheta = 2.*M_PI/nCore;
            for(size_t k = 0; k<nCore; ++k){
                double theta = k*dTheta;
                double x = (_R + r*cos(theta))*cPsi;
                double y = (_R + r*cos(theta))*sPsi;
                double z = r*sin(theta);
                double ax = -sPsi;
                double ay = cPsi;
                volume = 2.0*r0*_dPsi*(2.0*dTheta*j*_R*r0 + (12.0*gsl_pow_2(j)+1.0)*gsl_pow_2(r0)*(sin(theta + dTheta) - sin(theta))/3.0);
                gsl_matrix_set(_position,index,0,x);
                gsl_matrix_set(_position,index,1,y);
                gsl_matrix_set(_position,index,2,z);
                gsl_vector_set(_radius,index,sigma);
                gsl_vector_set(_volume,index,volume);
                gsl_vector_set(_birthstrength,index,1.0);
                index++;
            }
        }
    }
    OUT("Number of particles",index);
    set_vorticity();
}
*/

pawan::__vring::~__vring(){
}

void pawan::__vring::setStates(const gsl_vector *state){
	//OUT("setStates");
	gsl_vector_memcpy(_strength,state);
	set_vorticity();
}

void pawan::__vring::getRates(gsl_vector *rate){
	//OUT("getRates");
	for(size_t i = 0; i<_size; ++i){
		double ax = gsl_matrix_get(_vorticityfield,i,0);
		double ay = gsl_matrix_get(_vorticityfield,i,1);
		gsl_vector_set(rate,i,sqrt(ax*ax + ay*ay));
	}
}

void pawan::__vring::getStates(gsl_vector *state){
	//OUT("getStates");
	gsl_vector_memcpy(state,_strength);
}

void pawan::__vring::getIdealRates(gsl_vector *rate){
	//OUT("getIdealRates");
	size_t index = 0;
	double mag0 = 0.5/(M_PI*_sigmaR*_sigmaR);
    //double r0 = _sigmaR/(2*_nLayers + 1);		/*!< Central area */
    double r0 = 0.35/(2*_nLayers + 1);
	gsl_vector_set(rate,index,mag0);
	index++;
	for(size_t j = 1; j<=_nLayers; ++j){
		double r = r0*(12.*gsl_pow_2(j)+1.0)/(6.*j);
		size_t nCore = 8*j;
		double dTheta = 2.*M_PI/nCore;
		double decay = exp(-0.5*r*r/(_sigmaR*_sigmaR));
		for(size_t k = 0; k<nCore; ++k){
			double theta = k*dTheta;
			gsl_vector_set(rate,index,mag0*(1.0 + r*cos(theta)/_R)*decay);
			index++;
		}
	}
}

void pawan::__vring::print(){
	size_t indexV = 0;	
	// Central particle
	gsl_vector_view UR = gsl_matrix_row(_velocity,indexV);
	//OUTT("UR",&UR.vector);
	indexV++;
	for(size_t j = 1; j<=_nLayers; ++j){
		size_t nCore = 8*j;
		for(size_t k = 0; k<nCore; ++k){
			// Particles in layer
			gsl_vector_view UR = gsl_matrix_row(_velocity,indexV);
			//OUTT("UR",&UR.vector);
			indexV++;
		}
	}
}
