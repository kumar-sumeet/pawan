/*! PArticle Wake ANalysis
 * \file interaction.cpp
 * \brief Routines for interactions
 *
 * @author Puneet Singh
 * @date 04/24/2021
 */
#include "interaction.h"
#include "interaction_utils.h"

pawan::__interaction::__interaction(){
    DOUT("--------------------------------in pawan::__interaction::__interaction()");
    //_nu = 5.0e-3;   //vring fission-fusion
	//_nu = 2.5e-3;   //vring
    _nu = 1.56e-5;    //coupling
    //_nu = 0.0;
	_nWake = 0;
	_totalVorticity = gsl_vector_calloc(3);
	_linearImpulse = gsl_vector_calloc(3);
	_angularImpulse = gsl_vector_calloc(3);
    printf("_nu = %+8.3e\n",_nu);
}

pawan::__interaction::__interaction(__wake *W):__interaction(){
    DOUT("--------------------------------in pawan::__interaction::__interaction(__wake *W):__interaction()");
    //_maxsize = W->_maxsize;
    addWake(W);
}

pawan::__interaction::__interaction(__wake *W1, __wake *W2):__interaction(){
    DOUT("--------------------------------in pawan::__interaction::__interaction(__wake *W1, __wake *W2):__interaction()");
    //_maxsize = 2*W1->_maxsize; //max values same for both wakes
    addWake(W1);
	addWake(W2);
}

pawan::__interaction::~__interaction(){
    DOUT("--------------------------------in pawan::__interaction::~__interaction()");
	gsl_vector_free(_totalVorticity);
	gsl_vector_free(_linearImpulse);
	gsl_vector_free(_angularImpulse);
}

void pawan::__interaction::addWake(__wake *W){
    DOUT("--------------------------------in pawan::__interaction::addWake()");
	_W.push_back(W);
	_size = W->_size; //temp fix for Vring
    _totalmaxsize+= W->_maxsize;
	_nWake++;
    //printf("Position of 1st particle     interaction,addWake(): %+8.3e, %+8.3e, %+8.3e\n",
    //       gsl_matrix_get(W->_position, 0, 0),gsl_matrix_get(W->_position, 0, 1),gsl_matrix_get(W->_position, 0, 2));
}

void pawan::__interaction::solve(){
    DOUT("--------------------------------in pawan::__interaction::solve()");
	interact();
}

void pawan::__interaction::resolve(){
    DOUT("--------------------------------in pawan::__interaction::resolve()");
	influence();
}

void pawan::__interaction::diagnose(){
    DOUT("--------------------------------in pawan::__interaction::diagnose()");
	// Linear diagnostics
	gsl_vector_set_zero(_totalVorticity);
	gsl_vector_set_zero(_linearImpulse);
	gsl_vector_set_zero(_angularImpulse);

	gsl_vector *O = gsl_vector_calloc(3);
	gsl_vector *I = gsl_vector_calloc(3);
	gsl_vector *A = gsl_vector_calloc(3);
    for(auto &w : _W){
		// Total vorticity
		calculateTotalVorticity(w,O);
        gsl_vector_add(_totalVorticity,O);
		// Linear impulse
		calculateLinearImpulse(w,I);
        gsl_vector_add(_linearImpulse,I);
		// Angular impulse
		calculateAngularImpulse(w,A);
        gsl_vector_add(_angularImpulse,A);
	}
	OUTT("Total Vorticity",_totalVorticity);
	OUTT("Linear Impulse",_linearImpulse);
	OUTT("Angular Impulse",_angularImpulse);
	gsl_vector_free(O);
	gsl_vector_free(I);
	gsl_vector_free(A);

	// Quadratic diagnostics
	_helicity = 0.0;
	for(size_t i = 0; i<_nWake; ++i){
		_helicity += calculateHelicity(_W[i]);
		for(size_t j = i + 1; j < _nWake; ++j){
			_helicity += calculateHelicity(_W[i],_W[j]);
		}
	}
	OUT("Helicity",_helicity);
	_enstrophy = 0.0;
	for(size_t i = 0; i<_nWake; ++i){
		_enstrophy += calculateEnstrophy(_W[i]);
		for(size_t j = i + 1; j < _nWake; ++j){
			_enstrophy += calculateEnstrophy(_W[i],_W[j]);
		}
	}
	OUT("Enstrophy",_enstrophy);
    _enstrophyF = 0.0;
    for(size_t i = 0; i<_nWake; ++i){
        _enstrophyF += calculateEnstrophyF(_W[i]);
        for(size_t j = i + 1; j < _nWake; ++j){
            _enstrophyF += calculateEnstrophyF(_W[i],_W[j]);
        }
    }
    OUT("EnstrophyF",_enstrophyF);
    _kineticEnergy = 0.0;
    for(size_t i = 0; i<_nWake; ++i){
        _kineticEnergy += calculateKineticEnergy(_W[i]);
        for(size_t j = i + 1; j < _nWake; ++j){
            _kineticEnergy += calculateKineticEnergy(_W[i],_W[j]);
        }
    }
    OUT("KineticEnergy",_kineticEnergy);
    _kineticEnergyF = 0.0;
    for(size_t i = 0; i<_nWake; ++i){
        _kineticEnergyF += calculateKineticEnergyF(_W[i]);
        for(size_t j = i + 1; j < _nWake; ++j){
            _kineticEnergyF += calculateKineticEnergyF(_W[i],_W[j]);
        }
    }
    OUT("KineticEnergyF",_kineticEnergyF);


    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    double oy = 0.0;
    for(auto &w : _W) {
        double num = 0.0;
        double denom = 0.0;
        for (size_t i = 0; i < w->_numParticles; ++i) {
            gsl_vector_const_view ivor = gsl_matrix_const_row(w->_vorticity, i);
            oy = gsl_blas_dnrm2(&ivor.vector);
            x = gsl_matrix_get(w->_position, i, 0);
            y = gsl_matrix_get(w->_position, i, 1);
            z = gsl_matrix_get(w->_position, i, 2);
            num += oy * z * (x * x + y * y);
            denom += oy * (x * x + y * y);
            //num += oy * z;
            //denom += oy;
        }
        if (denom != 0.0) {
            _Zc = num / denom;
        }
        OUT("num ",num);
        OUT("denom ",denom);
    }
    OUT("Centroid of wake Zc",_Zc);

/* This is leading to a wrong result (WHY???)
    gsl_vector *rxo = gsl_vector_calloc(3);
    gsl_vector *Zc = gsl_vector_calloc(3);
    gsl_vector *dZc = gsl_vector_calloc(3);
    double rxoI = 0.0;
    double li = gsl_blas_dnrm2(_linearImpulse);
    for(auto &w : _W) {
        for (size_t i = 0; i < w->_numParticles; ++i) {
            gsl_vector_const_view ipos = gsl_matrix_const_row(w->_position, i);
            gsl_vector_const_view ivor = gsl_matrix_const_row(w->_vorticity, i);
            gsl_cross(&ipos.vector, &ivor.vector, rxo);
            gsl_blas_ddot(rxo, I, &rxoI);
            gsl_vector_memcpy(dZc, &ipos.vector);
            if (li != 0.0) {
                gsl_blas_dscal(0.5 * rxoI / li / li, dZc);
                gsl_vector_add(Zc, dZc);
            }
        }
        OUTT("Centroid of wake Zc", Zc);
        gsl_vector_free(rxo);
        gsl_vector_free(dZc);
        gsl_vector_free(Zc);
    }
*/

//vel at origin of ring
gsl_vector *vbi = gsl_vector_calloc(3);
gsl_vector *r = gsl_vector_calloc(3);
for (size_t k = 0; k < 3; ++k) {
    gsl_vector_set(r, k, 0.0);
}
getVi(r,vbi,0);
OUTT("vbi",vbi);
gsl_vector_free(vbi);
gsl_vector_free(r);
}

void pawan::__interaction::interact(){
    DOUT("--------------------------------in pawan::__interaction::interact()");
	for(auto &w : _W){
		gsl_matrix_set_zero(w->_velocity);
		gsl_matrix_set_zero(w->_retvorcity);
	}
	for(auto &w : _W){
		interact(w);
	}
	for(size_t i = 0; i<_nWake; ++i){
		for(size_t j = i + 1; j < _nWake; ++j){
			interact(_W[i],_W[j]);
		}
	}
}

void pawan::__interaction::influence(){
    DOUT("--------------------------------in pawan::__interaction::influence()");
	for(auto &w : _W){
		gsl_matrix_set_zero(w->_vorticityfield);
	}
	for(auto &w : _W){
		influence(w);
	}
	for(size_t i = 0; i<_nWake; ++i){
		for(size_t j = i + 1; j < _nWake; ++j){
			influence(_W[i],_W[j]);
		}
	}
}

void pawan::__interaction::influence(__wake *W){
    DOUT("--------------------------------in pawan::__interaction::influence(__wake *W)");
	for(size_t i_src = 0; i_src < W->_numParticles; ++i_src){
		gsl_vector_const_view r_src = gsl_matrix_const_row(W->_position,i_src);
		gsl_vector_const_view a_src = gsl_matrix_const_row(W->_vorticity,i_src);
		gsl_vector_view k_src = gsl_matrix_row(W->_vorticityfield,i_src);
		double s_src = gsl_vector_get(W->_radius,i_src);
		SELFINFLUENCE(s_src,&r_src.vector,&a_src.vector,&k_src.vector);
		for(size_t i_trg = i_src + 1; i_trg < W->_numParticles; ++i_trg){
			gsl_vector_const_view r_trg = gsl_matrix_const_row(W->_position,i_trg);
			gsl_vector_const_view a_trg= gsl_matrix_const_row(W->_vorticity,i_trg);
			gsl_vector_view k_trg = gsl_matrix_row(W->_vorticityfield,i_trg);
			double s_trg = gsl_vector_get(W->_radius,i_trg);
			INFLUENCE(s_src,s_trg,&r_src.vector,&r_trg.vector,&a_src.vector,&a_trg.vector,&k_src.vector,&k_trg.vector);
		}
	}
}

void pawan::__interaction::influence(__wake *W1, __wake *W2){
    DOUT("--------------------------------in pawan::__interaction::influence(__wake *W1, __wake *W2)");
	for(size_t i_src = 0; i_src < W1->_numParticles; ++i_src){
		gsl_vector_const_view r_src = gsl_matrix_const_row(W1->_position,i_src);
		gsl_vector_const_view a_src = gsl_matrix_const_row(W1->_vorticity,i_src);
		gsl_vector_view k_src = gsl_matrix_row(W1->_vorticityfield,i_src);
		double s_src = gsl_vector_get(W1->_radius,i_src);
		for(size_t i_trg = 0; i_trg < W2->_numParticles; ++i_trg){
			gsl_vector_const_view r_trg = gsl_matrix_const_row(W2->_position,i_trg);
			gsl_vector_const_view a_trg = gsl_matrix_const_row(W2->_vorticity,i_trg);
			gsl_vector_view k_trg = gsl_matrix_row(W2->_vorticityfield,i_trg);
			double s_trg = gsl_vector_get(W2->_radius,i_trg);
			INFLUENCE(s_src,s_trg,&r_src.vector,&r_trg.vector,&a_src.vector,&a_trg.vector,&k_src.vector,&k_trg.vector);
		}
	}
}

void pawan::__interaction::interact(__wake *W){
    DOUT("--------------------------------in pawan::__interaction::interact(__wake *W)");
	for(size_t i_src = 0; i_src < W->_numParticles; ++i_src){
		gsl_vector_const_view r_src = gsl_matrix_const_row(W->_position,i_src);
		gsl_vector_const_view a_src = gsl_matrix_const_row(W->_vorticity,i_src);
		gsl_vector_view dr_src = gsl_matrix_row(W->_velocity,i_src);
		gsl_vector_view da_src = gsl_matrix_row(W->_retvorcity,i_src);
		double s_src = gsl_vector_get(W->_radius,i_src);
		double v_src = gsl_vector_get(W->_volume,i_src);
		for(size_t i_trg = i_src + 1; i_trg < W->_numParticles; ++i_trg){
			gsl_vector_const_view r_trg = gsl_matrix_const_row(W->_position,i_trg);
			gsl_vector_const_view a_trg= gsl_matrix_const_row(W->_vorticity,i_trg);
			gsl_vector_view dr_trg = gsl_matrix_row(W->_velocity,i_trg);
			gsl_vector_view da_trg = gsl_matrix_row(W->_retvorcity,i_trg);
			double s_trg = gsl_vector_get(W->_radius,i_trg);
			double v_trg = gsl_vector_get(W->_volume,i_trg);
			INTERACT(_nu,s_src,s_trg,&r_src.vector,&r_trg.vector,&a_src.vector,&a_trg.vector,v_src,v_trg,&dr_src.vector,&dr_trg.vector,&da_src.vector,&da_trg.vector);
		}
	}
}

void pawan::__interaction::interact(__wake *W1, __wake *W2){
    DOUT("--------------------------------in pawan::__interaction::interact(__wake *W1, __wake *W2)");
	for(size_t i_src = 0; i_src < W1->_numParticles; ++i_src){
		gsl_vector_const_view r_src = gsl_matrix_const_row(W1->_position,i_src);
		gsl_vector_const_view a_src = gsl_matrix_const_row(W1->_vorticity,i_src);
		gsl_vector_view dr_src = gsl_matrix_row(W1->_velocity,i_src);
		gsl_vector_view da_src = gsl_matrix_row(W1->_retvorcity,i_src);
		double s_src = gsl_vector_get(W1->_radius,i_src);
		double v_src = gsl_vector_get(W1->_volume,i_src);
		for(size_t i_trg = 0; i_trg < W2->_numParticles; ++i_trg){
			gsl_vector_const_view r_trg = gsl_matrix_const_row(W2->_position,i_trg);
			gsl_vector_const_view a_trg = gsl_matrix_const_row(W2->_vorticity,i_trg);
			gsl_vector_view dr_trg = gsl_matrix_row(W2->_velocity,i_trg);
			gsl_vector_view da_trg = gsl_matrix_row(W2->_retvorcity,i_trg);
			double s_trg = gsl_vector_get(W2->_radius,i_trg);
			double v_trg = gsl_vector_get(W2->_volume,i_trg);
			INTERACT(_nu,s_src,s_trg,&r_src.vector,&r_trg.vector,&a_src.vector,&a_trg.vector,v_src,v_trg,&dr_src.vector,&dr_trg.vector,&da_src.vector,&da_trg.vector);
		}
	}
}

void pawan::__interaction::write(FILE *f){
    DOUT("--------------------------------in pawan::__interaction::write(FILE *f)");
	fwrite(&_nWake,sizeof(size_t),1,f);

    int write = 1;
	for(auto &w: _W){
	    if (write){ //max particles same for all wake, write only once
            int maxnumparticles = w->_maxnumParticles;
            fwrite(&maxnumparticles,sizeof(size_t),1,f);
	        write=0;
	    }
        //int maxnumparticles = w->_maxnumParticles;
        //fwrite(&maxnumparticles,sizeof(size_t),1,f);
		w->write(f);
	}
}

void pawan::__interaction::writenu(FILE *fdiag){
    fwrite(&_nu,sizeof(double),1,fdiag);
}

void pawan::__interaction::setDiagnostics(double *totalDiag){
    for(size_t i = 0; i<3; ++i) {
        gsl_vector_set(_totalVorticity, i, totalDiag[i]);
        gsl_vector_set(_linearImpulse, i, totalDiag[3+i]);
        gsl_vector_set(_angularImpulse, i, totalDiag[6+i]);
    }
    _enstrophy = totalDiag[9];
    _kineticEnergy = totalDiag[10];
    _helicity = totalDiag[11];
    _enstrophyF = totalDiag[12];
    _kineticEnergyF = totalDiag[13];
    if(totalDiag[15]!=0)
        _Zc = totalDiag[14]/totalDiag[15];
}

void pawan::__interaction::writediagnosis(FILE *fdiag){
    gsl_vector_fwrite(fdiag,_totalVorticity);
    gsl_vector_fwrite(fdiag,_linearImpulse);
    gsl_vector_fwrite(fdiag,_angularImpulse);
    fwrite(&_helicity,sizeof(double),1,fdiag);
    fwrite(&_enstrophy,sizeof(double),1,fdiag);
    fwrite(&_enstrophyF,sizeof(double),1,fdiag);
    fwrite(&_kineticEnergy,sizeof(double),1,fdiag);
    fwrite(&_kineticEnergyF,sizeof(double),1,fdiag);
    fwrite(&_Zc,sizeof(double),1,fdiag);
}

void pawan::__interaction::printdiagnostics(double *totalDiag){
    printf("\tTotal Vorticity = \t%21.16e %21.16e %21.16e \n", totalDiag[0], totalDiag[1], totalDiag[2]);
    printf("\tLinear Impulse  = \t%21.16e %21.16e %21.16e \n", totalDiag[3], totalDiag[4], totalDiag[5]);
    printf("\tAngular Impulse = \t%21.16e %21.16e %21.16e \n", totalDiag[6], totalDiag[7], totalDiag[8]);
    printf("Helicity            = %21.16e \n", totalDiag[11]);
    printf("Enstrophy           = %21.16e \n", totalDiag[9]);
    printf("EnstrophyF          = %21.16e \n", totalDiag[12]);
    printf("KineticEnergy       = %21.16e \n", totalDiag[10]);
    printf("KineticEnergyF      = %21.16e \n", totalDiag[13]);
    if(totalDiag[15]!=0) {
        printf("Centroid of wake Zc = %21.16e \n", totalDiag[14] / totalDiag[15]);
    }
    printf("	vbi = %21.16e, %21.16e, %21.16e \n", totalDiag[17],totalDiag[18],totalDiag[19]);
}

void pawan::__interaction::setStates(const gsl_vector *state){
    DOUT("--------------------------------in pawan::__interaction::setStates(const gsl_vector *state)");
	size_t offset = 0;
	for(auto &w: _W){
		gsl_vector_const_view st = gsl_vector_const_subvector(state,offset,w->_maxsize);
		w->setStates(&st.vector);
		offset += w->_maxsize;
	}
}

void pawan::__interaction::getRates(gsl_vector *rate){
    DOUT("--------------------------------in pawan::__interaction::getRates(gsl_vector *rate)");
	size_t offset = 0;
	for(auto &w: _W){
		gsl_vector_view rt = gsl_vector_subvector(rate,offset,w->_maxsize);
		w->getRates(&rt.vector);
		offset += w->_maxsize;
	}
}

void pawan::__interaction::getStates(gsl_vector *state){
    DOUT("--------------------------------in pawan::__interaction::getStates(gsl_vector *state)");
	size_t offset = 0;
	for(auto &w: _W){
		gsl_vector_view st = gsl_vector_subvector(state,offset,w->_maxsize);
		w->getStates(&st.vector);
		offset += w->_maxsize;
	}
}

void pawan::__interaction::split(size_t &stepnum){
    size_t offset = 0;
    for(auto &w: _W){
        w->split(stepnum);
        offset += w->_maxsize;
    }
}

void pawan::__interaction::merge(size_t &stepnum){
    size_t offset = 0;
    for(auto &w: _W){
        w->merge(stepnum);
        offset += w->_maxsize;
    }
}

void pawan::__interaction::addParticles(PawanRecvData pawanrecvdata,size_t &stepnum){
    size_t offset = 0;
    for(auto &w: _W){
        w->addParticles(pawanrecvdata,stepnum);
        offset += w->_maxsize;
    }
}

void pawan::__interaction::updateVinfEffect(const double *Vinf,double &dt){
    size_t offset = 0;
    for(auto &w: _W){
        w->updateVinfEffect(Vinf,dt);
        offset += w->_maxsize;
    }
}

void pawan::__interaction::updateBoundVorBoundVorEffectVind(PawanRecvData pawanrecvdata,PawanSendData pawansenddata,int astidx, int lfnidx){
    for(auto &w: _W){
        w->updateBoundVorBoundVorEffectVind(pawanrecvdata,pawansenddata,astidx, lfnidx);
    }
}
void pawan::__interaction::updateBoundVorEffect(PawanRecvData pawanrecvdata,double &dt,size_t &stepnum){
    size_t offset = 0;
    for(auto &w: _W){
        w->updateBoundVorEffect(pawanrecvdata,dt,stepnum); //effect of all lifting surfaces on each wake
        offset += w->_maxsize;
    }
}

void pawan::__interaction::getInflow(PawanRecvData pawanrecvdata, PawanSendData pawansenddata){
    int NbOfLfnLines = pawanrecvdata->NbOfLfnLines;
    int *NbOfAst = pawanrecvdata->NbOfAst;
    double *astpos = pawanrecvdata->astpos;

    double *lambda = pawansenddata->lambda;
    for (size_t j=0; j<PAWAN_MAXLFNLINES*PAWAN_MAXAST*3; j++){
        lambda[j]=0.0;
    }
    int astidx = 0;
    for (size_t ilfn = 0; ilfn < NbOfLfnLines; ++ilfn) {
        printf("------------------------------\n");
        for (size_t iast = 0; iast < NbOfAst[ilfn]; ++iast) {
            gsl_vector *vbi = gsl_vector_calloc(3);
            gsl_vector *rast = gsl_vector_calloc(3);
            for (size_t k = 0; k < 3; ++k) {
                gsl_vector_set(rast, k, astpos[astidx*3 + k]);
            }

            getVi(rast,vbi,iast);

            for (size_t k = 0; k < 3; ++k) {
                lambda[astidx*3 + k] = gsl_vector_get(vbi, k);
            }
/*            printf("---> lambda = %+10.5e, %+10.5e, %+10.5e @ast = %+10.5e, %+10.5e, %+10.5e  \n",
                   lambda[astidx*3],lambda[astidx*3 + 1],lambda[astidx*3 + 2],
                   astpos[astidx*3],astpos[astidx*3 + 1],astpos[astidx*3 + 2]);
*/          printf("CPU---> lambda = %+10.5e, %+10.5e, %+10.5e \n",
            lambda[astidx*3],lambda[astidx*3 + 1],lambda[astidx*3 + 2]);
            astidx++;
            gsl_vector_free(vbi);
            gsl_vector_free(rast);
        }
    }

    gsl_vector *vi = gsl_vector_calloc(3);
    gsl_vector *r = gsl_vector_calloc(3);
    gsl_vector_set(r,0,0);gsl_vector_set(r,1,0.8);gsl_vector_set(r,2,0);
    gsl_vector_set_zero(vi);
    getVi(r,vi,0);
    printf("Vi at Mid-wing ast = %10.5e, %10.5e, %10.5e \n",gsl_vector_get(vi,0),gsl_vector_get(vi,1),gsl_vector_get(vi,2));
    gsl_vector_set(r,0,-0.255);gsl_vector_set(r,1,0.8);gsl_vector_set(r,2,0.02);
    gsl_vector_set_zero(vi);
    getVi(r,vi,0);
    printf("Vi at Mid-wing TE  = %10.5e, %10.5e, %10.5e \n",gsl_vector_get(vi,0),gsl_vector_get(vi,1),gsl_vector_get(vi,2));
    gsl_vector_set(r,0,-2.5);gsl_vector_set(r,1,0.8);gsl_vector_set(r,2,0.02);
    gsl_vector_set_zero(vi);
    getVi(r,vi,0);
    printf("Vi at point 2 = %10.5e, %10.5e, %10.5e \n",gsl_vector_get(vi,0),gsl_vector_get(vi,1),gsl_vector_get(vi,2));
    gsl_vector_set(r,0,0.0);gsl_vector_set(r,1,1.6);gsl_vector_set(r,2,0.0);
    gsl_vector_set_zero(vi);
    getVi(r,vi,0);
    printf("Vi at point 3 = %10.5e, %10.5e, %10.5e \n",gsl_vector_get(vi,0),gsl_vector_get(vi,1),gsl_vector_get(vi,2));

    gsl_vector_free(vi);gsl_vector_free(r);

}

void pawan::__interaction::getVi(const gsl_vector *r, gsl_vector *vi, const size_t &n){
    for (auto &W: _W) {//induced inflow due to each wake
        for (size_t i = 0; i < W->_numParticles; ++i) {
            gsl_vector *displacement = gsl_vector_calloc(3);
            gsl_vector_const_view ipos = gsl_matrix_const_row(W->_position, i);
            gsl_vector_const_view ivor = gsl_matrix_const_row(W->_vorticity, i);
            double sigma = gsl_vector_get(W->_radius, i);

            gsl_vector_memcpy(displacement, r);
            gsl_vector_sub(displacement, &ipos.vector);
            double rho = gsl_blas_dnrm2(displacement);
            double q = 0.0;
            q = QSIG(rho,sigma);
            //if(i%17==0 && n==8)
                //printf("q = %10.5e \t",q);
            // Velocity computation
            gsl_vector *dv = gsl_vector_calloc(3);
            VELOCITY(-q, &ivor.vector, displacement, dv);
            gsl_vector_add(vi, dv);

            gsl_vector_free(displacement);
            gsl_vector_free(dv);
        }
    }
}

void pawan::__interaction::getIdealRates(gsl_vector *rate){
    DOUT("--------------------------------in pawan::__interaction::getIdealRates(gsl_vector *rate)");
	size_t offset = 0;
	for(auto &w: _W){
		gsl_vector_view rt = gsl_vector_subvector(rate,offset,w->_maxsize);
		w->getIdealRates(&rt.vector);
		offset += w->_maxsize;
	}
}

void pawan::__interaction::calculateTotalVorticity(__wake *W, gsl_vector *O){
    DOUT("--------------------------------in pawan::__interaction::calculateTotalVorticity(__wake *W, gsl_vector *O)");
	gsl_vector_set_zero(O); //ip: redundant, *O was already set as calloc
	for(size_t i = 0; i < W->_numParticles; ++i){
		gsl_vector_const_view a = gsl_matrix_const_row(W->_vorticity,i);
		gsl_vector_add(O,&a.vector);
	}
}

void pawan::__interaction::calculateLinearImpulse(__wake *W, gsl_vector *I){
    DOUT("--------------------------------in pawan::__interaction::calculateLinearImpulse(__wake *W, gsl_vector *I)");
	gsl_vector_set_zero(I);//ip: redundant
	gsl_vector *rxa = gsl_vector_calloc(3);
	for(size_t i = 0; i < W->_numParticles; ++i){
		gsl_vector_const_view r = gsl_matrix_const_row(W->_position,i);
		gsl_vector_const_view a = gsl_matrix_const_row(W->_vorticity,i);
		gsl_cross(&r.vector,&a.vector,rxa);
		gsl_vector_add(I,rxa);
	}
	gsl_vector_scale(I,0.5);
	gsl_vector_free(rxa);
}

void pawan::__interaction::calculateAngularImpulse(__wake *W, gsl_vector *A){
    DOUT("--------------------------------in pawan::__interaction::calculateAngularImpulse(__wake *W, gsl_vector *A)");
	gsl_vector_set_zero(A);//ip: redundant
	gsl_vector *rxa = gsl_vector_calloc(3);
	gsl_vector *rxrxa = gsl_vector_calloc(3);
	for(size_t i = 0; i < W->_numParticles; ++i){
		gsl_vector_const_view r = gsl_matrix_const_row(W->_position,i);
		gsl_vector_const_view a = gsl_matrix_const_row(W->_vorticity,i);
		gsl_cross(&r.vector,&a.vector,rxa);
		gsl_cross(&r.vector,rxa,rxrxa);
		gsl_vector_add(A,rxrxa);
	}
	gsl_vector_scale(A,1.0/3.0);
	gsl_vector_free(rxrxa);
	gsl_vector_free(rxa);
}

double pawan::__interaction::calculateEnstrophy(__wake *W){
    DOUT("--------------------------------in pawan::__interaction::calculateEnstrophy(__wake *W)");
	double s = 0.0;
	for(size_t I = 0; I < W->_numParticles; ++I){
		gsl_vector_const_view rI = gsl_matrix_const_row(W->_position,I);
		gsl_vector_const_view aI = gsl_matrix_const_row(W->_vorticity,I);
		double sI = gsl_vector_get(W->_radius,I);
		s += ENSTROPHY(sI,&aI.vector);
		//double ens = ENSTROPHY(sI,&aI.vector);
		//s += ens;
		//OUT("e",ens);
		for(size_t J = I + 1; J < W->_numParticles; ++J){
			//OUT("\tJ",J);
			gsl_vector_const_view rJ = gsl_matrix_const_row(W->_position,J);
			gsl_vector_const_view aJ = gsl_matrix_const_row(W->_vorticity,J);
			double sJ = gsl_vector_get(W->_radius,J);
			// doubling S because S(i,j) = S(j,i)
			s += 2*ENSTROPHY(sI,sJ,&rI.vector,&rJ.vector,&aI.vector,&aJ.vector);
			//OUT("\ts",s);
		}
	}
	return s;
}

double pawan::__interaction::calculateEnstrophy(__wake *W1, __wake *W2){
	double s = 0.0;
	for(size_t I = 0; I < W1->_numParticles; ++I){
		gsl_vector_const_view rI = gsl_matrix_const_row(W1->_position,I);
		gsl_vector_const_view aI = gsl_matrix_const_row(W1->_vorticity,I);
		double sI = gsl_vector_get(W1->_radius,I);
		for(size_t J = 0; J < W2->_numParticles; ++J){
			gsl_vector_const_view rJ = gsl_matrix_const_row(W2->_position,J);
			gsl_vector_const_view aJ = gsl_matrix_const_row(W2->_vorticity,J);
			double sJ = gsl_vector_get(W2->_radius,J);
			s += 2.0*ENSTROPHY(sI,sJ,&rI.vector,&rJ.vector,&aI.vector,&aJ.vector);
		}
	}
	// doubling S because S(i,j) = S(j,i)
	return s;
}

double pawan::__interaction::calculateEnstrophyF(__wake *W){
    DOUT("--------------------------------in pawan::__interaction::calculateEnstrophyF(__wake *W)");
    double s = 0.0;
    for(size_t I = 0; I < W->_numParticles; ++I){
        gsl_vector_const_view rI = gsl_matrix_const_row(W->_position,I);
        gsl_vector_const_view aI = gsl_matrix_const_row(W->_vorticity,I);
        double sI = gsl_vector_get(W->_radius,I);
        s += ENSTROPHYF(sI,&aI.vector);
        for(size_t J = I + 1; J < W->_numParticles; ++J){
            //OUT("\tJ",J);
            gsl_vector_const_view rJ = gsl_matrix_const_row(W->_position,J);
            gsl_vector_const_view aJ = gsl_matrix_const_row(W->_vorticity,J);
            double sJ = gsl_vector_get(W->_radius,J);
            // doubling S because S(i,j) = S(j,i)
            s += 2*ENSTROPHYF(sI,sJ,&rI.vector,&rJ.vector,&aI.vector,&aJ.vector);
            //OUT("\ts",s);
        }
    }
    return s;
}

double pawan::__interaction::calculateEnstrophyF(__wake *W1, __wake *W2){
    double s = 0.0;
    for(size_t I = 0; I < W1->_numParticles; ++I){
        gsl_vector_const_view rI = gsl_matrix_const_row(W1->_position,I);
        gsl_vector_const_view aI = gsl_matrix_const_row(W1->_vorticity,I);
        double sI = gsl_vector_get(W1->_radius,I);
        for(size_t J = 0; J < W2->_numParticles; ++J){
            gsl_vector_const_view rJ = gsl_matrix_const_row(W2->_position,J);
            gsl_vector_const_view aJ = gsl_matrix_const_row(W2->_vorticity,J);
            double sJ = gsl_vector_get(W2->_radius,J);
            s += 2.0*ENSTROPHYF(sI,sJ,&rI.vector,&rJ.vector,&aI.vector,&aJ.vector);
        }
    }
    // doubling S because S(i,j) = S(j,i)
    return s;
}

double pawan::__interaction::calculateHelicity(__wake *W){
    DOUT("--------------------------------in pawan::__interaction::calculateHelicity(__wake *W)");
	int test_counter=0;
    double h = 0.0;
	for(size_t I = 0; I < W->_numParticles; ++I){
		gsl_vector_const_view rI = gsl_matrix_const_row(W->_position,I);
		gsl_vector_const_view aI = gsl_matrix_const_row(W->_vorticity,I);
		double sI = gsl_vector_get(W->_radius,I);
		for(size_t J = I + 1; J < W->_numParticles; ++J){
			gsl_vector_const_view rJ = gsl_matrix_const_row(W->_position,J);
			gsl_vector_const_view aJ = gsl_matrix_const_row(W->_vorticity,J);
			double sJ = gsl_vector_get(W->_radius,J);
            test_counter = test_counter+1;
			//printf("I=%d, J=%d ==== %d",I,J,test_counter);
            //printf("number of particles %d",W->_numParticles);
            h += 2.0*HELICITY(sI,sJ,&rI.vector,&rJ.vector,&aI.vector,&aJ.vector);
		}
	}
	// doubling H because H(i,j) = H(j,i)
	return h;
}

double pawan::__interaction::calculateHelicity(__wake *W1, __wake *W2){
    DOUT("--------------------------------in pawan::__interaction::calculateHelicity(__wake *W1, __wake *W2)");
	double h = 0.0;
	for(size_t I = 0; I < W1->_numParticles; ++I){
		gsl_vector_const_view rI = gsl_matrix_const_row(W1->_position,I);
		gsl_vector_const_view aI = gsl_matrix_const_row(W1->_vorticity,I);
		double sI = gsl_vector_get(W1->_radius,I);
		for(size_t J = 0; J < W2->_numParticles; ++J){
			gsl_vector_const_view rJ = gsl_matrix_const_row(W2->_position,J);
			gsl_vector_const_view aJ = gsl_matrix_const_row(W2->_vorticity,J);
			double sJ = gsl_vector_get(W2->_radius,J);
			h += 2.0*HELICITY(sI,sJ,&rI.vector,&rJ.vector,&aI.vector,&aJ.vector);
		}
	}
	// doubling H because H(i,j) = H(j,i)
	return h;
}

double pawan::__interaction::calculateKineticEnergy(__wake *W){
    DOUT("--------------------------------in pawan::__interaction::calculateKineticEnergy(__wake *W)");
	double ke = 0.0;
	for(size_t I = 0; I < W->_numParticles; ++I){
		gsl_vector_const_view rI = gsl_matrix_const_row(W->_position,I);
		gsl_vector_const_view aI = gsl_matrix_const_row(W->_vorticity,I);
		double sI = gsl_vector_get(W->_radius,I);
		ke += KINETICENERGY(sI,&aI.vector);
		for(size_t J = I + 1; J < W->_numParticles; ++J){
			gsl_vector_const_view rJ = gsl_matrix_const_row(W->_position,J);
			gsl_vector_const_view aJ = gsl_matrix_const_row(W->_vorticity,J);
			double sJ = gsl_vector_get(W->_radius,J);
			ke += 2.0*KINETICENERGY(sI,sJ,&rI.vector,&rJ.vector,&aI.vector,&aJ.vector);
		}
	}
	return ke;
}

double pawan::__interaction::calculateKineticEnergy(__wake *W1, __wake *W2){
    DOUT("--------------------------------in pawan::__interaction::calculateKineticEnergy(__wake *W1, __wake *W2)");
	double ke = 0.0;
	for(size_t I = 0; I < W1->_numParticles; ++I){
		gsl_vector_const_view rI = gsl_matrix_const_row(W1->_position,I);
		gsl_vector_const_view aI = gsl_matrix_const_row(W1->_vorticity,I);
		double sI = gsl_vector_get(W1->_radius,I);
		for(size_t J = 0; J < W2->_numParticles; ++J){
			gsl_vector_const_view rJ = gsl_matrix_const_row(W2->_position,J);
			gsl_vector_const_view aJ = gsl_matrix_const_row(W2->_vorticity,J);
			double sJ = gsl_vector_get(W2->_radius,J);
			ke += 2.0*KINETICENERGY(sI,sJ,&rI.vector,&rJ.vector,&aI.vector,&aJ.vector);
		}
	}
	// doubling KE because KE(i,j) = KE(j,i)
	return ke;
}

double pawan::__interaction::calculateKineticEnergyF(__wake *W){
    DOUT("--------------------------------in pawan::__interaction::calculateKineticEnergy(__wake *W)");
    double ke = 0.0;
    for(size_t I = 0; I < W->_numParticles; ++I){
        gsl_vector_const_view rI = gsl_matrix_const_row(W->_position,I);
        gsl_vector_const_view aI = gsl_matrix_const_row(W->_vorticity,I);
        double sI = gsl_vector_get(W->_radius,I);
        ke += KINETICENERGYF(sI,&aI.vector);
        for(size_t J = I + 1; J < W->_numParticles; ++J){
            gsl_vector_const_view rJ = gsl_matrix_const_row(W->_position,J);
            gsl_vector_const_view aJ = gsl_matrix_const_row(W->_vorticity,J);
            double sJ = gsl_vector_get(W->_radius,J);
            ke += 2.0*KINETICENERGYF(sI,sJ,&rI.vector,&rJ.vector,&aI.vector,&aJ.vector);
        }
    }
    return ke;
}

double pawan::__interaction::calculateKineticEnergyF(__wake *W1, __wake *W2){
    DOUT("--------------------------------in pawan::__interaction::calculateKineticEnergy(__wake *W1, __wake *W2)");
    double ke = 0.0;
    for(size_t I = 0; I < W1->_numParticles; ++I){
        gsl_vector_const_view rI = gsl_matrix_const_row(W1->_position,I);
        gsl_vector_const_view aI = gsl_matrix_const_row(W1->_vorticity,I);
        double sI = gsl_vector_get(W1->_radius,I);
        for(size_t J = 0; J < W2->_numParticles; ++J){
            gsl_vector_const_view rJ = gsl_matrix_const_row(W2->_position,J);
            gsl_vector_const_view aJ = gsl_matrix_const_row(W2->_vorticity,J);
            double sJ = gsl_vector_get(W2->_radius,J);
            ke += 2.0*KINETICENERGYF(sI,sJ,&rI.vector,&rJ.vector,&aI.vector,&aJ.vector);
        }
    }
    // doubling KE because KE(i,j) = KE(j,i)
    return ke;
}

void pawan::__interaction::setParticles(double *p){
    auto rateBuffer = reinterpret_cast<double4*>(p);

    int index = 0;

    for(auto w : _W){
        for(int i = 0; i < w->_numParticles; i++, index++){
            gsl_matrix_set(w->_position,i,0,rateBuffer[2 * index].x);
            gsl_matrix_set(w->_position,i,1,rateBuffer[2 * index].y);
            gsl_matrix_set(w->_position,i,2,rateBuffer[2 * index].z);
            gsl_matrix_set(w->_vorticity,i,0,rateBuffer[2 * index + 1].x);
            gsl_matrix_set(w->_vorticity,i,1,rateBuffer[2 * index + 1].y);
            gsl_matrix_set(w->_vorticity,i,2,rateBuffer[2 * index + 1].z);
            //ignore volume and smoothing radius because they should not have changed
        }
    }

}

void pawan::__interaction::getParticles(double *p,int *age, size_t stepnum){

    auto particlesBuffer = reinterpret_cast<double4*>(p);

    int position = 0;

    for(auto const w : _W){
        for(int i = 0; i < w->_numParticles; i++, position++) {
            particlesBuffer[2 * position].x = gsl_matrix_get(w->_position, i, 0);
            particlesBuffer[2 * position].y = gsl_matrix_get(w->_position, i, 1);
            particlesBuffer[2 * position].z = gsl_matrix_get(w->_position, i, 2);
            particlesBuffer[2 * position].w = gsl_vector_get(w->_radius, i);
            particlesBuffer[2 * position + 1].x = gsl_matrix_get(w->_vorticity, i, 0);
            particlesBuffer[2 * position + 1].y = gsl_matrix_get(w->_vorticity, i, 1);
            particlesBuffer[2 * position + 1].z = gsl_matrix_get(w->_vorticity, i, 2);
            particlesBuffer[2 * position + 1].w = gsl_vector_get(w->_volume, i);
            age[position] = stepnum - gsl_vector_get(w->_active, i);
        }
    }
}

void pawan::__interaction::getParticles_arr(double *p){

    int position = 0;

    int totalparticles = amountParticles();

    for(auto const w : _W){
        for(int i = 0; i < w->_numParticles; i++, position++) {
            p[                   position] = gsl_matrix_get(w->_position, i, 0);
            p[  totalparticles + position] = gsl_matrix_get(w->_position, i, 1);
            p[2*totalparticles + position] = gsl_matrix_get(w->_position, i, 2);
            p[3*totalparticles + position] = gsl_vector_get(w->_radius,   i);
            p[4*totalparticles + position] = gsl_matrix_get(w->_vorticity, i, 0);
            p[5*totalparticles + position] = gsl_matrix_get(w->_vorticity, i, 1);
            p[6*totalparticles + position] = gsl_matrix_get(w->_vorticity, i, 2);
            p[7*totalparticles + position] = gsl_vector_get(w->_volume,    i);
            p[8*totalparticles + position] = sqrt( gsl_matrix_get(w->_vorticity, i, 0)*gsl_matrix_get(w->_vorticity, i, 0)
                                                  +gsl_matrix_get(w->_vorticity, i, 1)*gsl_matrix_get(w->_vorticity, i, 1)
                                                  +gsl_matrix_get(w->_vorticity, i, 2)*gsl_matrix_get(w->_vorticity, i, 2)
                                                 );
        }
    }

}

    int pawan::__interaction::amountParticles() {
    int total = 0;
    for(auto const w : _W){
        total += w->_numParticles;
    }
    return total;
}

int pawan::__interaction::totalmaxParticles() {
    int totalmaxparticles = 0;
    for(auto const w : _W){
        totalmaxparticles += w->_maxnumParticles;
    }
    return totalmaxparticles;
}

double pawan::__interaction::getNu() {
    return _nu;
}
