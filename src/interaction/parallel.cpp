/*! PArticle Wake ANalysis
 * \file parallel.cpp
 * \brief Routines for Open MP parallilized interactions
 *
 * @author Puneet Singh
 * @date 04/24/2021
 */
#include "parallel.h"
#include "src/utils/print_utils.h"

pawan::__parallel::__parallel(__wake *W):__interaction(W){}
pawan::__parallel::__parallel(__wake *W1, __wake *W2):__interaction(W1,W2){}

void pawan::__parallel::interact(__wake *W){
	for(size_t i_src = 0; i_src < W->_numParticles; ++i_src){
		gsl_vector_const_view r_src = gsl_matrix_const_row(W->_position,i_src);
		gsl_vector_const_view a_src = gsl_matrix_const_row(W->_vorticity,i_src);
		double s_src = gsl_vector_get(W->_radius,i_src);
		double v_src = gsl_vector_get(W->_volume,i_src);
		double vx = 0.0, vy = 0.0, vz = 0.0;
		double qx = 0.0, qy = 0.0, qz = 0.0;
		#pragma omp parallel for reduction(+:vx,vy,vz,qx,qy,qz)
		for(size_t i_trg = i_src + 1; i_trg < W->_numParticles; ++i_trg){
			gsl_vector_const_view r_trg = gsl_matrix_const_row(W->_position,i_trg);
			gsl_vector_const_view a_trg= gsl_matrix_const_row(W->_vorticity,i_trg);
			gsl_vector_view dr_trg = gsl_matrix_row(W->_velocity,i_trg);
			gsl_vector_view da_trg = gsl_matrix_row(W->_retvorcity,i_trg);
			double s_trg = gsl_vector_get(W->_radius,i_trg);
			double v_trg = gsl_vector_get(W->_volume,i_trg);
			double vx_s = 0.0, vy_s = 0.0, vz_s = 0.0;
			double qx_s = 0.0, qy_s = 0.0, qz_s = 0.0;
			INTERACT(_nu,s_src,s_trg,&r_src.vector,&r_trg.vector,&a_src.vector,&a_trg.vector,v_src,v_trg,&dr_trg.vector,&da_trg.vector,vx_s,vy_s,vz_s,qx_s,qy_s,qz_s);
			vx += vx_s;
			vy += vy_s;
			vz += vz_s;
			qx += qx_s;
			qy += qy_s;
			qz += qz_s;
		}
		gsl_vector_view dr_src = gsl_matrix_row(W->_velocity,i_src);
		gsl_vector_set(&dr_src.vector,0,vx + gsl_vector_get(&dr_src.vector,0));
		gsl_vector_set(&dr_src.vector,1,vy + gsl_vector_get(&dr_src.vector,1));
		gsl_vector_set(&dr_src.vector,2,vz + gsl_vector_get(&dr_src.vector,2));
		gsl_vector_view da_src = gsl_matrix_row(W->_retvorcity,i_src);
		gsl_vector_set(&da_src.vector,0,qx + gsl_vector_get(&da_src.vector,0));
		gsl_vector_set(&da_src.vector,1,qy + gsl_vector_get(&da_src.vector,1));
		gsl_vector_set(&da_src.vector,2,qz + gsl_vector_get(&da_src.vector,2));
	}
}

void pawan::__parallel::interact(__wake *W1, __wake *W2){
	for(size_t i_src = 0; i_src < W1->_numParticles; ++i_src){
		gsl_vector_const_view r_src = gsl_matrix_const_row(W1->_position,i_src);
		gsl_vector_const_view a_src = gsl_matrix_const_row(W1->_vorticity,i_src);
		double s_src = gsl_vector_get(W1->_radius,i_src);
		double v_src = gsl_vector_get(W1->_volume,i_src);
		double vx = 0.0, vy = 0.0, vz = 0.0;
		double qx = 0.0, qy = 0.0, qz = 0.0;
		#pragma omp parallel for reduction(+:vx,vy,vz,qx,qy,qz)
		for(size_t i_trg = 0; i_trg < W2->_numParticles; ++i_trg){
			gsl_vector_const_view r_trg = gsl_matrix_const_row(W2->_position,i_trg);
			gsl_vector_const_view a_trg = gsl_matrix_const_row(W2->_vorticity,i_trg);
			gsl_vector_view dr_trg = gsl_matrix_row(W2->_velocity,i_trg);
			gsl_vector_view da_trg = gsl_matrix_row(W2->_retvorcity,i_trg);
			double s_trg = gsl_vector_get(W2->_radius,i_trg);
			double v_trg = gsl_vector_get(W2->_volume,i_trg);
			double vx_s = 0.0, vy_s = 0.0, vz_s = 0.0;
			double qx_s = 0.0, qy_s = 0.0, qz_s = 0.0;
			INTERACT(_nu,s_src,s_trg,&r_src.vector,&r_trg.vector,&a_src.vector,&a_trg.vector,v_src,v_trg,&dr_trg.vector,&da_trg.vector,vx_s,vy_s,vz_s,qx_s,qy_s,qz_s);
			vx += vx_s;
			vy += vy_s;
			vz += vz_s;
			qx += qx_s;
			qy += qy_s;
			qz += qz_s;
		}
		gsl_vector_view dr_src = gsl_matrix_row(W1->_velocity,i_src);
		gsl_vector_view da_src = gsl_matrix_row(W1->_retvorcity,i_src);
		gsl_vector_set(&dr_src.vector,0,vx + gsl_vector_get(&dr_src.vector,0));
		gsl_vector_set(&dr_src.vector,1,vy + gsl_vector_get(&dr_src.vector,1));
		gsl_vector_set(&dr_src.vector,2,vz + gsl_vector_get(&dr_src.vector,2));
		gsl_vector_set(&da_src.vector,0,qx + gsl_vector_get(&da_src.vector,0));
		gsl_vector_set(&da_src.vector,1,qy + gsl_vector_get(&da_src.vector,1));
		gsl_vector_set(&da_src.vector,2,qz + gsl_vector_get(&da_src.vector,2));
	}
}

void pawan::__parallel::influence(__wake *W){
	for(size_t i_src = 0; i_src < W->_numParticles; ++i_src){
		gsl_vector_const_view r_src = gsl_matrix_const_row(W->_position,i_src);
		gsl_vector_const_view a_src = gsl_matrix_const_row(W->_vorticity,i_src);
		double s_src = gsl_vector_get(W->_radius,i_src);
		gsl_vector_view k_src = gsl_matrix_row(W->_vorticityfield,i_src);
		SELFINFLUENCE(s_src,&r_src.vector,&a_src.vector,&k_src.vector);
		double kx = 0.0, ky = 0.0, kz = 0.0;
		#pragma omp parallel for reduction(+:kx,ky,kz)
		for(size_t i_trg = i_src + 1; i_trg < W->_numParticles; ++i_trg){
			gsl_vector_const_view r_trg = gsl_matrix_const_row(W->_position,i_trg);
			gsl_vector_const_view a_trg= gsl_matrix_const_row(W->_vorticity,i_trg);
			gsl_vector_view k_trg = gsl_matrix_row(W->_vorticityfield,i_trg);
			double s_trg = gsl_vector_get(W->_radius,i_trg);
			double kx_s = 0.0, ky_s = 0.0, kz_s = 0.0;
			INFLUENCE(s_src,s_trg,&r_src.vector,&r_trg.vector,&a_src.vector,&a_trg.vector,&k_trg.vector,kx_s,ky_s,kz_s);
			kx += kx_s;
			ky += ky_s;
			kz += kz_s;
		}
		gsl_vector_set(&k_src.vector,0,kx + gsl_vector_get(&k_src.vector,0));
		gsl_vector_set(&k_src.vector,1,ky + gsl_vector_get(&k_src.vector,1));
		gsl_vector_set(&k_src.vector,2,kz + gsl_vector_get(&k_src.vector,2));
	}
}

void pawan::__parallel::influence(__wake *W1, __wake *W2){
	for(size_t i_src = 0; i_src < W1->_numParticles; ++i_src){
		gsl_vector_const_view r_src = gsl_matrix_const_row(W1->_position,i_src);
		gsl_vector_const_view a_src = gsl_matrix_const_row(W1->_vorticity,i_src);
		double s_src = gsl_vector_get(W1->_radius,i_src);
		double kx = 0.0, ky = 0.0, kz = 0.0;
		#pragma omp parallel for reduction(+:kx,ky,kz)
		for(size_t i_trg = 0; i_trg < W2->_numParticles; ++i_trg){
			gsl_vector_const_view r_trg = gsl_matrix_const_row(W2->_position,i_trg);
			gsl_vector_const_view a_trg = gsl_matrix_const_row(W2->_vorticity,i_trg);
			gsl_vector_view k_trg = gsl_matrix_row(W2->_vorticityfield,i_trg);
			double s_trg = gsl_vector_get(W2->_radius,i_trg);
			double v_trg = gsl_vector_get(W2->_volume,i_trg);
			double kx_s = 0.0, ky_s = 0.0, kz_s = 0.0;
			INFLUENCE(s_src,s_trg,&r_src.vector,&r_trg.vector,&a_src.vector,&a_trg.vector,&k_trg.vector,kx_s,ky_s,kz_s);
			kx += kx_s;
			ky += ky_s;
			kz += kz_s;
		}
		gsl_vector_view k_src = gsl_matrix_row(W1->_vorticityfield,i_src);
		gsl_vector_set(&k_src.vector,0,kx + gsl_vector_get(&k_src.vector,0));
		gsl_vector_set(&k_src.vector,1,ky + gsl_vector_get(&k_src.vector,1));
		gsl_vector_set(&k_src.vector,2,kz + gsl_vector_get(&k_src.vector,2));
	}
}

void pawan::__parallel::calculateLinearImpulse(__wake *W, gsl_vector *I){
	gsl_vector_set_zero(I);
	double Ix = 0.0, Iy = 0.0, Iz = 0.0;
	#pragma omp parallel for reduction(+:Ix,Iy,Iz)
	for(size_t i = 0; i < W->_numParticles; ++i){
		gsl_vector_const_view r = gsl_matrix_const_row(W->_position,i);
		gsl_vector_const_view a = gsl_matrix_const_row(W->_vorticity,i);
		gsl_vector *rxa = gsl_vector_calloc(3);
		gsl_cross(&r.vector,&a.vector,rxa);
		Ix += gsl_vector_get(rxa,0);
		Iy += gsl_vector_get(rxa,1);
		Iz += gsl_vector_get(rxa,2);
		gsl_vector_free(rxa);
	}
	gsl_vector_set(I,0,Ix);
	gsl_vector_set(I,1,Iy);
	gsl_vector_set(I,2,Iz);
	gsl_vector_scale(I,0.5);
}

void pawan::__parallel::calculateTotalVorticity(__wake *W, gsl_vector *O){
	gsl_vector_set_zero(O);
	double Ox = 0.0, Oy = 0.0, Oz = 0.0;
	#pragma omp parallel for reduction(+:Ox,Oy,Oz)
	for(size_t i = 0; i < W->_numParticles; ++i){
		Ox += gsl_matrix_get(W->_vorticity,i,0);
		Oy += gsl_matrix_get(W->_vorticity,i,1);
		Oz += gsl_matrix_get(W->_vorticity,i,2);
	}
	gsl_vector_set(O,0,Ox);
	gsl_vector_set(O,1,Oy);
	gsl_vector_set(O,2,Oz);
}

void pawan::__parallel::calculateAngularImpulse(__wake *W, gsl_vector *A){
	gsl_vector_set_zero(A);
	double Lx = 0.0, Ly = 0.0, Lz = 0.0;
	#pragma omp parallel for reduction(+:Lx,Ly,Lz)
	for(size_t i = 0; i < W->_numParticles; ++i){
		gsl_vector *rxa = gsl_vector_calloc(3);
		gsl_vector *rxrxa = gsl_vector_calloc(3);
		gsl_vector_const_view r = gsl_matrix_const_row(W->_position,i);
		gsl_vector_const_view a = gsl_matrix_const_row(W->_vorticity,i);
		gsl_cross(&r.vector,&a.vector,rxa);
		gsl_cross(&r.vector,rxa,rxrxa);
		Lx += gsl_vector_get(rxrxa,0);
		Ly += gsl_vector_get(rxrxa,1);
		Lz += gsl_vector_get(rxrxa,2);
		gsl_vector_free(rxrxa);
		gsl_vector_free(rxa);
	}
	gsl_vector_set(A,0,Lx/3.0);
	gsl_vector_set(A,1,Ly/3.0);
	gsl_vector_set(A,2,Lz/3.0);
}

double pawan::__parallel::calculateKineticEnergy(__wake *W){
	double ke = 0.0;
	for(size_t I = 0; I < W->_numParticles; ++I){
		gsl_vector_const_view rI = gsl_matrix_const_row(W->_position,I);
		gsl_vector_const_view aI = gsl_matrix_const_row(W->_vorticity,I);
		double sI = gsl_vector_get(W->_radius,I);
		ke += KINETICENERGY(sI,&aI.vector);
		double keJ = 0.0;
		#pragma omp parallel for reduction(+:keJ)
		for(size_t J = I + 1; J < W->_numParticles; ++J){
			gsl_vector_const_view rJ = gsl_matrix_const_row(W->_position,J);
			gsl_vector_const_view aJ = gsl_matrix_const_row(W->_vorticity,J);
			double sJ = gsl_vector_get(W->_radius,J);
			keJ += 2.0*KINETICENERGY(sI,sJ,&rI.vector,&rJ.vector,&aI.vector,&aJ.vector);
		}
		ke += keJ;
	}
	return ke;
}

double pawan::__parallel::calculateKineticEnergy(__wake *W1, __wake *W2){
	double ke = 0.0;
	for(size_t I = 0; I < W1->_numParticles; ++I){
		gsl_vector_const_view rI = gsl_matrix_const_row(W1->_position,I);
		gsl_vector_const_view aI = gsl_matrix_const_row(W1->_vorticity,I);
		double sI = gsl_vector_get(W1->_radius,I);
		double keJ = 0.0;
		#pragma omp parallel for reduction(+:keJ)
		for(size_t J = 0; J < W2->_numParticles; ++J){
			gsl_vector_const_view rJ = gsl_matrix_const_row(W2->_position,J);
			gsl_vector_const_view aJ = gsl_matrix_const_row(W2->_vorticity,J);
			double sJ = gsl_vector_get(W2->_radius,J);
			keJ += 2.0*KINETICENERGY(sI,sJ,&rI.vector,&rJ.vector,&aI.vector,&aJ.vector);
		}
		ke += keJ;
	}
	// doubling KE because KE(i,j) = KE(j,i)
	return ke;
}

double pawan::__parallel::calculateKineticEnergyF(__wake *W){
    double ke = 0.0;
    for(size_t I = 0; I < W->_numParticles; ++I){
        gsl_vector_const_view rI = gsl_matrix_const_row(W->_position,I);
        gsl_vector_const_view aI = gsl_matrix_const_row(W->_vorticity,I);
        double sI = gsl_vector_get(W->_radius,I);
        ke += KINETICENERGYF(sI,&aI.vector);
        double keJ = 0.0;
#pragma omp parallel for reduction(+:keJ)
        for(size_t J = I + 1; J < W->_numParticles; ++J){
            gsl_vector_const_view rJ = gsl_matrix_const_row(W->_position,J);
            gsl_vector_const_view aJ = gsl_matrix_const_row(W->_vorticity,J);
            double sJ = gsl_vector_get(W->_radius,J);
            keJ += 2.0*KINETICENERGYF(sI,sJ,&rI.vector,&rJ.vector,&aI.vector,&aJ.vector);
        }
        ke += keJ;
    }
    return ke;
}

double pawan::__parallel::calculateKineticEnergyF(__wake *W1, __wake *W2){
    double ke = 0.0;
    for(size_t I = 0; I < W1->_numParticles; ++I){
        gsl_vector_const_view rI = gsl_matrix_const_row(W1->_position,I);
        gsl_vector_const_view aI = gsl_matrix_const_row(W1->_vorticity,I);
        double sI = gsl_vector_get(W1->_radius,I);
        double keJ = 0.0;
#pragma omp parallel for reduction(+:keJ)
        for(size_t J = 0; J < W2->_numParticles; ++J){
            gsl_vector_const_view rJ = gsl_matrix_const_row(W2->_position,J);
            gsl_vector_const_view aJ = gsl_matrix_const_row(W2->_vorticity,J);
            double sJ = gsl_vector_get(W2->_radius,J);
            // doubling KE because KE(i,j) = KE(j,i)
            keJ += 2.0*KINETICENERGYF(sI,sJ,&rI.vector,&rJ.vector,&aI.vector,&aJ.vector);
        }
        ke += keJ;
    }

    return ke;
}

double pawan::__parallel::calculateHelicity(__wake *W){
	double h = 0.0;
	for(size_t I = 0; I < W->_numParticles; ++I){
		gsl_vector_const_view rI = gsl_matrix_const_row(W->_position,I);
		gsl_vector_const_view aI = gsl_matrix_const_row(W->_vorticity,I);
		double sI = gsl_vector_get(W->_radius,I);
		double hJ = 0.0;
		#pragma omp parallel for reduction(+:hJ)
		for(size_t J = I + 1; J < W->_numParticles; ++J){
			gsl_vector_const_view rJ = gsl_matrix_const_row(W->_position,J);
			gsl_vector_const_view aJ = gsl_matrix_const_row(W->_vorticity,J);
			double sJ = gsl_vector_get(W->_radius,J);
            // doubling H because H(i,j) = H(j,i)
			hJ += 2.0*HELICITY(sI,sJ,&rI.vector,&rJ.vector,&aI.vector,&aJ.vector);
		}
		h += hJ;
	}

	return h;
}

double pawan::__parallel::calculateHelicity(__wake *W1, __wake *W2){
	double h = 0.0;
	for(size_t I = 0; I < W1->_numParticles; ++I){
		gsl_vector_const_view rI = gsl_matrix_const_row(W1->_position,I);
		gsl_vector_const_view aI = gsl_matrix_const_row(W1->_vorticity,I);
		double sI = gsl_vector_get(W1->_radius,I);
		double hJ = 0.0;
		#pragma omp parallel for reduction(+:hJ)
		for(size_t J = 0; J < W2->_numParticles; ++J){
			gsl_vector_const_view rJ = gsl_matrix_const_row(W2->_position,J);
			gsl_vector_const_view aJ = gsl_matrix_const_row(W2->_vorticity,J);
			double sJ = gsl_vector_get(W2->_radius,J);
            // doubling H because H(i,j) = H(j,i)
			hJ += 2.0*HELICITY(sI,sJ,&rI.vector,&rJ.vector,&aI.vector,&aJ.vector);
		}
		h += hJ;
	}

	return h;
}

double pawan::__parallel::calculateEnstrophy(__wake *W){
	double s = 0.0;
	for(size_t I = 0; I < W->_numParticles; ++I){
		gsl_vector_const_view rI = gsl_matrix_const_row(W->_position,I);
		gsl_vector_const_view aI = gsl_matrix_const_row(W->_vorticity,I);
		double sI = gsl_vector_get(W->_radius,I);
		s += ENSTROPHY(sI,&aI.vector);
		double enJ = 0.0;
		#pragma omp parallel for reduction(+:enJ)
		for(size_t J = I + 1; J < W->_numParticles; ++J){
			gsl_vector_const_view rJ = gsl_matrix_const_row(W->_position,J);
			gsl_vector_const_view aJ = gsl_matrix_const_row(W->_vorticity,J);
			double sJ = gsl_vector_get(W->_radius,J);
			// doubling S because S(i,j) = S(j,i)
			enJ += 2*ENSTROPHY(sI,sJ,&rI.vector,&rJ.vector,&aI.vector,&aJ.vector);
		}
		s += enJ;
	}
	return s;
}

double pawan::__parallel::calculateEnstrophy(__wake *W1, __wake *W2){
	double s = 0.0;
	for(size_t I = 0; I < W1->_numParticles; ++I){
		gsl_vector_const_view rI = gsl_matrix_const_row(W1->_position,I);
		gsl_vector_const_view aI = gsl_matrix_const_row(W1->_vorticity,I);
		double sI = gsl_vector_get(W1->_radius,I);
		double enJ = 0.0;
		#pragma omp parallel for reduction(+:enJ)
		for(size_t J = 0; J < W2->_numParticles; ++J){
			gsl_vector_const_view rJ = gsl_matrix_const_row(W2->_position,J);
			gsl_vector_const_view aJ = gsl_matrix_const_row(W2->_vorticity,J);
			double sJ = gsl_vector_get(W2->_radius,J);
            // doubling S because S(i,j) = S(j,i)
			enJ += 2.0*ENSTROPHY(sI,sJ,&rI.vector,&rJ.vector,&aI.vector,&aJ.vector);
		}
		s += enJ;
	}
	return s;
}


double pawan::__parallel::calculateEnstrophyF(__wake *W){
    double s = 0.0;
    for(size_t I = 0; I < W->_numParticles; ++I){
        gsl_vector_const_view rI = gsl_matrix_const_row(W->_position,I);
        gsl_vector_const_view aI = gsl_matrix_const_row(W->_vorticity,I);
        double sI = gsl_vector_get(W->_radius,I);
        s += ENSTROPHYF(sI,&aI.vector);
        double enJ = 0.0;
#pragma omp parallel for reduction(+:enJ)
        for(size_t J = I + 1; J < W->_numParticles; ++J){
            gsl_vector_const_view rJ = gsl_matrix_const_row(W->_position,J);
            gsl_vector_const_view aJ = gsl_matrix_const_row(W->_vorticity,J);
            double sJ = gsl_vector_get(W->_radius,J);
            // doubling S because S(i,j) = S(j,i)
            enJ += 2*ENSTROPHYF(sI,sJ,&rI.vector,&rJ.vector,&aI.vector,&aJ.vector);
        }
        s += enJ;
    }
    return s;
}

double pawan::__parallel::calculateEnstrophyF(__wake *W1, __wake *W2){
    double s = 0.0;
    for(size_t I = 0; I < W1->_numParticles; ++I){
        gsl_vector_const_view rI = gsl_matrix_const_row(W1->_position,I);
        gsl_vector_const_view aI = gsl_matrix_const_row(W1->_vorticity,I);
        double sI = gsl_vector_get(W1->_radius,I);
        double enJ = 0.0;
#pragma omp parallel for reduction(+:enJ)
        for(size_t J = 0; J < W2->_numParticles; ++J){
            gsl_vector_const_view rJ = gsl_matrix_const_row(W2->_position,J);
            gsl_vector_const_view aJ = gsl_matrix_const_row(W2->_vorticity,J);
            double sJ = gsl_vector_get(W2->_radius,J);
            // doubling S because S(i,j) = S(j,i)
            enJ += 2.0*ENSTROPHYF(sI,sJ,&rI.vector,&rJ.vector,&aI.vector,&aJ.vector);
        }
        s += enJ;
    }

    return s;
}
