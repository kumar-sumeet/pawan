/*! PArticle Wake ANalysis
 * \file wake.cpp
 * \brief Routines for Wake calculations
 *
 * @author Puneet Singh
 * @date 03/28/2021
 */
#include "wake.h"

void pawan::__wake::create_particles(const int &n){
	_numParticles = n;
	_size = 2*_numParticles*3;
	_position = gsl_matrix_calloc(_numParticles,3);
	_velocity = gsl_matrix_calloc(_numParticles,3);
	_vorticity = gsl_matrix_calloc(_numParticles,3);
	_retvorcity = gsl_matrix_calloc(_numParticles,3);
	_radius = gsl_vector_calloc(_numParticles);
	_volume = gsl_vector_calloc(_numParticles);
	_birthstrength = gsl_vector_calloc(_numParticles);
	_vorticityfield = gsl_matrix_calloc(_numParticles,3);
}

pawan::__wake::__wake(){
	create_particles(1);
	for(int i = 0; i<_numParticles; ++i){
		for(int j = 0; j<3; ++j){
			gsl_matrix_set(_position,i,j,0.0);
			gsl_matrix_set(_vorticity,i,j,0.0);
			gsl_matrix_set(_vorticityfield,i,j,0.0);
		}
		gsl_vector_set(_radius,i,1.0);
		gsl_vector_set(_volume,i,1.0);
		gsl_vector_set(_birthstrength,i,1.0);
	}
}

pawan::__wake::~__wake(){
	gsl_matrix_free(_position);
	gsl_matrix_free(_velocity);
	gsl_matrix_free(_vorticity);
	gsl_matrix_free(_retvorcity);
	gsl_vector_free(_radius);
	gsl_vector_free(_volume);
	gsl_vector_free(_birthstrength);
	gsl_matrix_free(_vorticityfield);
}

void pawan::__wake::print(){
	OUT("_numParticles",_numParticles);
	OUT("_position",_position);
	OUT("_vorticity",_vorticity);
	OUT("_radius",_radius);
	OUT("_volume",_volume);
	OUT("_birthstrength",_birthstrength);
	OUT("_vorticityfield",_vorticityfield);
}

void pawan::__wake::save(FILE *f){
	fwrite(&_numParticles,sizeof(size_t),1,f);	
	gsl_matrix_fwrite(f,_position);
	gsl_matrix_fwrite(f,_vorticity);
	gsl_vector_fwrite(f,_radius);
	gsl_vector_fwrite(f,_volume);
	gsl_vector_fwrite(f,_birthstrength);
	//gsl_matrix_fwrite(f,_vorticityfield);
}

void pawan::__wake::write(FILE *f){
	fwrite(&_numParticles,sizeof(size_t),1,f);	
	gsl_matrix_fwrite(f,_position);
	gsl_matrix_fwrite(f,_vorticity);
	gsl_vector_fwrite(f,_radius);
}

void pawan::__wake::read(__io *IO){
	FILE *f = IO->open_binary_file(".wake");
	fread(&_numParticles,sizeof(size_t),1,f);	
	gsl_matrix_fread(f,_position);
	gsl_matrix_fread(f,_vorticity);
	gsl_vector_fread(f,_radius);
	gsl_vector_fread(f,_volume);
	gsl_vector_fread(f,_birthstrength);
	//gsl_matrix_fread(f,_vorticityfield);
	fclose(f);
}

void pawan::__wake::setStates(const gsl_vector *state){
	//OUT("setStates");
	size_t np = state->size/2/3;
	size_t matrixsize = np*3;
	gsl_matrix_const_view pos = gsl_matrix_const_view_vector(state,np,3);
	gsl_matrix_memcpy(_position,&pos.matrix);
	gsl_vector_const_view vor = gsl_vector_const_subvector(state,matrixsize,matrixsize);
	gsl_matrix_const_view vrx = gsl_matrix_const_view_vector(&vor.vector,np,3);
	gsl_matrix_memcpy(_vorticity,&vrx.matrix);
}

void pawan::__wake::getRates(gsl_vector *rate){
	//OUT("getRates");
	for(size_t i = 0; i<_numParticles; ++i){
		for(size_t j = 0; j<3; ++j){
			size_t ind = i*3 + j;
			gsl_vector_set(rate,ind,gsl_matrix_get(_velocity,i,j));
			ind += _size/2;
			gsl_vector_set(rate,ind,gsl_matrix_get(_retvorcity,i,j));
		}
	}
}

void pawan::__wake::getStates(gsl_vector *state){
	//OUT("getStates");
	for(size_t i = 0; i<_numParticles; ++i){
		for(size_t j = 0; j<3; ++j){
			size_t ind = i*3 + j;
			gsl_vector_set(state,ind,gsl_matrix_get(_position,i,j));
			ind += _size/2;
			gsl_vector_set(state,ind,gsl_matrix_get(_vorticity,i,j));
		}
	}
}

void pawan::__wake::translate(const size_t &n, const double &x){
	for(int i = 0; i<_numParticles; ++i){
		gsl_matrix_set(_position,i,n,x + gsl_matrix_get(_position,i,n));
	}

}
