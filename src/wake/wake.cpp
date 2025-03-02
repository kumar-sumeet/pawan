/*! PArticle Wake ANalysis
 * \file wake.cpp
 * \brief Routines for Wake calculations
 *
 * @author Puneet Singh
 * @date 03/28/2021
 */
#include "wake.h"

void pawan::__wake::create_particles(const int &n){
    DOUT("------------------------------------------------------in pawan::__wake::create_particles()");
	_numParticles    = n;
	_size            = 2*_numParticles*3;
    _maxnumParticles = _numParticles;
    _maxsize         = 2*_maxnumParticles*3;
	_position        = gsl_matrix_calloc(_numParticles,3);
	_velocity        = gsl_matrix_calloc(_numParticles,3);
	_vorticity       = gsl_matrix_calloc(_numParticles,3);
	_retvorcity      = gsl_matrix_calloc(_numParticles,3);
	_radius          = gsl_vector_calloc(_numParticles);
	_volume          = gsl_vector_calloc(_numParticles);
	_birthstrength   = gsl_vector_calloc(_numParticles);
	_vorticityfield  = gsl_matrix_calloc(_numParticles,3);
    _active          = gsl_vector_calloc(_numParticles);
}

void pawan::__wake::initialise_memory(){
    DOUT("------------------------------------------------------in pawan::__wake::create_particles()");
    _npidx  = 0;
    _maxmem = false;
    _numParticles = 0; //no particle info in arrays yet
    _size = 2*_numParticles*3; //this becomes irrelevant in the case of large memory allocation
    _maxnumParticles = MAXNUMPARTICLES;
    _maxsize         = 2*_maxnumParticles*3;
    _position        = gsl_matrix_calloc(_maxnumParticles,3);
    _velocity        = gsl_matrix_calloc(_maxnumParticles,3);
    _vorticity       = gsl_matrix_calloc(_maxnumParticles,3);
    _retvorcity      = gsl_matrix_calloc(_maxnumParticles,3);
    _radius          = gsl_vector_calloc(_maxnumParticles);
    _volume          = gsl_vector_calloc(_maxnumParticles);
    _birthstrength   = gsl_vector_calloc(_maxnumParticles);
    _vorticityfield  = gsl_matrix_calloc(_maxnumParticles,3);
    _active          = gsl_vector_calloc(_maxnumParticles);
}

pawan::__wake::__wake(){
    DOUT("------------------------------------------------------in pawan::__wake::__wake()");
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

void pawan::__wake::split(size_t &stepnum){

    //split particles only after a certain age
    for(size_t i = 0; i<_numParticles && stepnum-gsl_vector_get(_active,i)>1; ++i){
        double ibirthstrength = gsl_vector_get(_birthstrength,i);
        gsl_vector_view ivor = gsl_matrix_row(_vorticity,i);
        double ivormag = gsl_blas_dnrm2(&ivor.vector);

        //can this be parallelised when all threads could be changing
        // _vorticity and _position at the same time?
        if(ivormag>=2*ibirthstrength && ibirthstrength>0.0001) {//ip: hard-coded, a more appropriate condition needs to be found
            printf("splitting!!!--------- %d --------- ", i);
            gsl_vector_view ipos = gsl_matrix_row(_position,i);
            double iradius = gsl_vector_get(_radius,i);
            double ivol = gsl_vector_get(_volume,i);
            gsl_vector *deltapos = gsl_vector_calloc(3);
            gsl_vector_memcpy(deltapos,&ivor.vector);
            gsl_blas_dscal(0.25*iradius/ivormag,deltapos);

            //add a particle
            gsl_vector_view ipos_split = gsl_matrix_row(_position,_npidx);
            gsl_vector_memcpy(&ipos_split.vector,&ipos.vector);
            gsl_vector_add(&ipos_split.vector,deltapos);
            gsl_vector_view ivor_split = gsl_matrix_row(_vorticity,_npidx);
            gsl_vector_memcpy(&ivor_split.vector,&ivor.vector);
            gsl_blas_dscal(0.5,&ivor_split.vector);
            gsl_vector_set(_radius,_npidx,iradius);
            gsl_vector_set(_volume,_npidx,ivol);  //need to confirm this
            gsl_vector_set(_birthstrength,_npidx,0.5*ivormag);
            gsl_vector_set(_active,_npidx,gsl_vector_get(_active,i));  //split particle age same as parent particle

            //replace particle parameters in-place
            gsl_vector_sub(&ipos.vector,deltapos);
            gsl_blas_dscal(0.5,&ivor.vector);
            gsl_vector_set(_birthstrength,i,0.5*ivormag);

            gsl_vector_free(deltapos);
            _npidx++;
            if(_npidx==MAXNUMPARTICLES){
                _npidx = 0;
                _numParticles = MAXNUMPARTICLES;
                _maxmem = true;
            }
            if(!_maxmem)
                _numParticles = _npidx;
        }
    }
    printf("\n");
    //_numParticles = _numParticles + npsplit;
    _size = 2*_numParticles*3;
}

void pawan::__wake::merge(size_t &stepnum){

    int npmerge = 0;//number of particles 'lost' due to merging
    for(size_t i = 0; i<_numParticles && stepnum-gsl_vector_get(_active,i)>45; ++i){//merge particles only after a certain age
        gsl_vector_view ipos = gsl_matrix_row(_position,i);
        gsl_vector_view ivor = gsl_matrix_row(_vorticity,i);
        double ivormag = gsl_blas_dnrm2(&ivor.vector);
        double iradius = gsl_vector_get(_radius,i);

        for(size_t j = i + 1; j < _numParticles; ++j) { //not checking particle age here since the check above
                                                        // takes care of it
            gsl_vector_view jpos = gsl_matrix_row(_position,j);
            gsl_vector_view jvor = gsl_matrix_row(_vorticity,j);
            double jvormag = gsl_blas_dnrm2(&jvor.vector);
            double jradius = gsl_vector_get(_radius,j);

            gsl_vector *displacement = gsl_vector_calloc(3);
            gsl_vector_memcpy(displacement,&ipos.vector);
            gsl_vector_sub(displacement,&jpos.vector);
            double rho = gsl_blas_dnrm2(displacement);
            if (rho<0.125*(iradius+jradius) && jvormag!=0.0 && ivormag!=0.0) {
                double alignment=0.0;
                gsl_blas_ddot(&ivor.vector,&jvor.vector,&alignment);
                alignment = 1 - alignment/ivormag/jvormag;
                if(alignment>=0.0001){//DBL_EPSILON??
                    printf("merging!!!--------- %d ----and---- %d--------- ", i,j);
                    gsl_vector *ipos_m = gsl_vector_calloc(3);
                    gsl_vector *tmp = gsl_vector_calloc(3);
                    gsl_vector_memcpy(ipos_m,&ipos.vector);
                    gsl_blas_dscal(ivormag,ipos_m);
                    gsl_vector_memcpy(tmp,&jpos.vector);
                    gsl_blas_dscal(jvormag,tmp);
                    gsl_vector_add(ipos_m,tmp);
                    gsl_blas_dscal(1/(ivormag+jvormag),ipos_m);
                    gsl_vector_memcpy(&ipos.vector,ipos_m);    //in-place replacement
                    gsl_vector_add(&ivor.vector,&jvor.vector);

                    gsl_vector_set_zero(&jvor.vector);     //"deleting" the other particle
                    gsl_vector_set(_volume,j,0.0);

                    gsl_vector_free(ipos_m);
                    gsl_vector_free(tmp);
                }
            }
            gsl_vector_free(displacement);
        }
    }
    printf("\n");
}

void pawan::__wake::getlfnvec(double* vec,
                             const double* mat,
                             const int rowsize,
                             const int axis,
                             const int offsetrows,
                             const int size){
    int offset = offsetrows*rowsize;
    for(size_t i = 0; i<size; ++i){
        vec[i] = mat[offset + rowsize * i + axis];
    }
}
/*!
void pawan::__wake::addParticlesold(PawanRecvData pawanrecvdata,size_t &stepnum){
    OUT("_numParticles",_numParticles);
    OUT("_npidx",_npidx);
    double *Vinf = pawanrecvdata->Vinf;
    int NbOfLfnLines = pawanrecvdata->NbOfLfnLines;
    int *NbOfAst = pawanrecvdata->NbOfAst;
    double *lfnlen = pawanrecvdata->lfnlen;
    int trailvor = pawanrecvdata->trailvor;
    int shedvor = pawanrecvdata->shedvor;
    int suprootvor = pawanrecvdata->suprootvor;
    int transientsteps = pawanrecvdata->transientsteps;
    //int infl2D = pawanrecvdata->infl2D;
    //int spanres = pawanrecvdata->spanres;
    int spanres = 30;  //number of particles over the span of a lfnline
    double acrossa = pawanrecvdata->acrossa;
    double deltat = pawanrecvdata->deltat;
    double *spandisc = pawanrecvdata->span_disc;
    double *TEpos = pawanrecvdata->TEpos;
    double *circ = pawanrecvdata->circ;
    double *TEpos_prev = pawanrecvdata->TEpos_prev;
    double *circ_prev = pawanrecvdata->circ_prev;
    double *astpos = pawanrecvdata->astpos;
    double *astvel = pawanrecvdata->astvel;

    double Vinfmag = sqrt(Vinf[0]*Vinf[0] + Vinf[1]*Vinf[1] + Vinf[2]*Vinf[2]);
    int astidx = 0;
    int offsetrows=0;
    int sectidx = 0; //index of section where particles originate (varies on lfnline from 0 to spanres; spanres+1 sections)
    int iast_peakcirc_lfn;
    double peakcirc_lfn, spandisc_peakcirc_lfn;
    double sigma, hres, alphai, trailvormag;
    bool dontaddTEpart;
    double tepos[3],teposprev[3], lastpart[3]; // _, _, coordinates of last particle to emanate from a section ista
    double del_teposmag, vx, vy, vz, vmag, voli;
    double localflowdir[3], trailvorpos[3];  //need more astkin details to get localflowdir accurately for blades
    int num_part; //
    double shedvorpos[3],shedvordir[3],shedvormag,shedvordirmag;
    double teposnxt[3], teposprevnxt[3], tepos_shed[3],teposprev_shed[3];
    double lastpart_shed[3]; //coordinates of last particle to emanate between ista and ista + 1
    for(size_t ilfn = 0; ilfn<NbOfLfnLines; ++ilfn) {

        double *TEposx_lfn = (double *) calloc(NbOfAst[ilfn], sizeof(double));
        double *TEposy_lfn = (double *) calloc(NbOfAst[ilfn], sizeof(double));
        double *TEposz_lfn = (double *) calloc(NbOfAst[ilfn], sizeof(double));
        getlfnvec(TEposx_lfn, TEpos, 3, 0, offsetrows, NbOfAst[ilfn]);
        getlfnvec(TEposy_lfn, TEpos, 3, 1, offsetrows, NbOfAst[ilfn]);
        getlfnvec(TEposz_lfn, TEpos, 3, 2, offsetrows, NbOfAst[ilfn]);
        double *TEposxprev_lfn = (double *) calloc(NbOfAst[ilfn], sizeof(double));
        double *TEposyprev_lfn = (double *) calloc(NbOfAst[ilfn], sizeof(double));
        double *TEposzprev_lfn = (double *) calloc(NbOfAst[ilfn], sizeof(double));
        getlfnvec(TEposxprev_lfn, TEpos_prev, 3, 0, offsetrows, NbOfAst[ilfn]);
        getlfnvec(TEposyprev_lfn, TEpos_prev, 3, 1, offsetrows, NbOfAst[ilfn]);
        getlfnvec(TEposzprev_lfn, TEpos_prev, 3, 2, offsetrows, NbOfAst[ilfn]);
        double *circ_lfn = (double *) calloc(NbOfAst[ilfn], sizeof(double));
        getlfnvec(circ_lfn, circ, 1, 0, offsetrows, NbOfAst[ilfn]);
        double *circprev_lfn = (double *) calloc(NbOfAst[ilfn], sizeof(double));
        getlfnvec(circprev_lfn, circ_prev, 1, 0, offsetrows, NbOfAst[ilfn]);
        double *astposx_lfn = (double *) calloc(NbOfAst[ilfn], sizeof(double));
        double *astposy_lfn = (double *) calloc(NbOfAst[ilfn], sizeof(double));
        double *astposz_lfn = (double *) calloc(NbOfAst[ilfn], sizeof(double));
        getlfnvec(astposx_lfn, astpos, 3, 0, offsetrows, NbOfAst[ilfn]);
        getlfnvec(astposy_lfn, astpos, 3, 1, offsetrows, NbOfAst[ilfn]);
        getlfnvec(astposz_lfn, astpos, 3, 2, offsetrows, NbOfAst[ilfn]);
        double *astvelx_lfn = (double *) calloc(NbOfAst[ilfn], sizeof(double));
        double *astvely_lfn = (double *) calloc(NbOfAst[ilfn], sizeof(double));
        double *astvelz_lfn = (double *) calloc(NbOfAst[ilfn], sizeof(double));
        getlfnvec(astvelx_lfn, astvel, 3, 0, offsetrows, NbOfAst[ilfn]);
        getlfnvec(astvely_lfn, astvel, 3, 1, offsetrows, NbOfAst[ilfn]);
        getlfnvec(astvelz_lfn, astvel, 3, 2, offsetrows, NbOfAst[ilfn]);
        double *spandisc_lfn = (double *) calloc(NbOfAst[ilfn], sizeof(double));
        getlfnvec(spandisc_lfn, spandisc, 1, 0, offsetrows, NbOfAst[ilfn]);
        offsetrows += NbOfAst[ilfn];//also just astidx can be used

        gsl_interp_accel *accel_ptr;
        gsl_spline *circ_spline, *TEposx_spline, *TEposy_spline, *TEposz_spline;
        gsl_spline *circprev_spline, *TEposxprev_spline, *TEposyprev_spline, *TEposzprev_spline;
        gsl_spline *astposx_spline, *astposy_spline, *astposz_spline;
        gsl_spline *astvelx_spline, *astvely_spline, *astvelz_spline;
        accel_ptr = gsl_interp_accel_alloc();
        circ_spline = gsl_spline_alloc(gsl_interp_cspline, NbOfAst[ilfn]);
        TEposx_spline = gsl_spline_alloc(gsl_interp_cspline, NbOfAst[ilfn]);
        TEposy_spline = gsl_spline_alloc(gsl_interp_cspline, NbOfAst[ilfn]);
        TEposz_spline = gsl_spline_alloc(gsl_interp_cspline, NbOfAst[ilfn]);
        circprev_spline = gsl_spline_alloc(gsl_interp_cspline, NbOfAst[ilfn]);
        TEposxprev_spline = gsl_spline_alloc(gsl_interp_cspline, NbOfAst[ilfn]);
        TEposyprev_spline = gsl_spline_alloc(gsl_interp_cspline, NbOfAst[ilfn]);
        TEposzprev_spline = gsl_spline_alloc(gsl_interp_cspline, NbOfAst[ilfn]);
        astposx_spline = gsl_spline_alloc(gsl_interp_cspline, NbOfAst[ilfn]);
        astposy_spline = gsl_spline_alloc(gsl_interp_cspline, NbOfAst[ilfn]);
        astposz_spline = gsl_spline_alloc(gsl_interp_cspline, NbOfAst[ilfn]);
        astvelx_spline = gsl_spline_alloc(gsl_interp_cspline, NbOfAst[ilfn]);
        astvely_spline = gsl_spline_alloc(gsl_interp_cspline, NbOfAst[ilfn]);
        astvelz_spline = gsl_spline_alloc(gsl_interp_cspline, NbOfAst[ilfn]);
        gsl_spline_init(circ_spline, spandisc_lfn, circ_lfn, NbOfAst[ilfn]);
        gsl_spline_init(TEposx_spline, spandisc_lfn, TEposx_lfn, NbOfAst[ilfn]);
        gsl_spline_init(TEposy_spline, spandisc_lfn, TEposy_lfn, NbOfAst[ilfn]);
        gsl_spline_init(TEposz_spline, spandisc_lfn, TEposz_lfn, NbOfAst[ilfn]);
        gsl_spline_init(circprev_spline, spandisc_lfn, circprev_lfn, NbOfAst[ilfn]);
        gsl_spline_init(TEposxprev_spline, spandisc_lfn, TEposxprev_lfn, NbOfAst[ilfn]);
        gsl_spline_init(TEposyprev_spline, spandisc_lfn, TEposyprev_lfn, NbOfAst[ilfn]);
        gsl_spline_init(TEposzprev_spline, spandisc_lfn, TEposzprev_lfn, NbOfAst[ilfn]);
        gsl_spline_init(astposx_spline, spandisc_lfn, astposx_lfn, NbOfAst[ilfn]);
        gsl_spline_init(astposy_spline, spandisc_lfn, astposy_lfn, NbOfAst[ilfn]);
        gsl_spline_init(astposz_spline, spandisc_lfn, astposz_lfn, NbOfAst[ilfn]);
        gsl_spline_init(astvelx_spline, spandisc_lfn, astvelx_lfn, NbOfAst[ilfn]);
        gsl_spline_init(astvely_spline, spandisc_lfn, astvely_lfn, NbOfAst[ilfn]);
        gsl_spline_init(astvelz_spline, spandisc_lfn, astvelz_lfn, NbOfAst[ilfn]);

        //get peak circulation and index
        iast_peakcirc_lfn = 0;          //airsta index where peak circulation occurs
        peakcirc_lfn = circ_lfn[0];  //peak lfnline circulation
        spandisc_peakcirc_lfn = 0.0; //non-dim radius where peak circ occurs
        for (int i = 1; i < NbOfAst[ilfn]; i++) {
            if (circ[i] > peakcirc_lfn) {
                peakcirc_lfn = circ_lfn[i];
                iast_peakcirc_lfn = i;
                spandisc_peakcirc_lfn = spandisc_lfn[i];
            }
        }

        if(shedvor && trailvor)
            hres =  lfnlen[ilfn] / (2 * spanres - 1);
        else
            hres = lfnlen[ilfn] / spanres;
        sigma = 1.3 * hres;

        for (int ista = 0; ista < spanres + 1; ++ista) {//new discretized station index
            sectidx = ilfn * (2 * spanres + 1) + 2 * ista;
            dontaddTEpart = true;  //whether to add particles right at the TE

            tepos[0] = gsl_spline_eval(TEposx_spline, ista * (1. / spanres), accel_ptr);
            tepos[1] = gsl_spline_eval(TEposy_spline, ista * (1. / spanres), accel_ptr);
            tepos[2] = gsl_spline_eval(TEposz_spline, ista * (1. / spanres), accel_ptr);

            teposprev[0] = gsl_spline_eval(TEposxprev_spline, ista * (1. / spanres), accel_ptr);
            teposprev[1] = gsl_spline_eval(TEposyprev_spline, ista * (1. / spanres), accel_ptr);
            teposprev[2] = gsl_spline_eval(TEposzprev_spline, ista * (1. / spanres), accel_ptr);

            if (stepnum == 0) {
                gsl_matrix_set(_initTEpos, sectidx, 0, teposprev[0]);
                gsl_matrix_set(_initTEpos, sectidx, 1, teposprev[1]);
                gsl_matrix_set(_initTEpos, sectidx, 2, teposprev[2]);
                dontaddTEpart = false; //add particle right at the TE only for the first time step
            }
            if (gsl_vector_get(_lastsectpartnpidx, sectidx)!=DBL_MAX) {
                lastpart[0] = gsl_matrix_get(_position, gsl_vector_get(_lastsectpartnpidx, sectidx), 0);
                lastpart[1] = gsl_matrix_get(_position, gsl_vector_get(_lastsectpartnpidx, sectidx), 1);
                lastpart[2] = gsl_matrix_get(_position, gsl_vector_get(_lastsectpartnpidx, sectidx), 2);
            }
            else{
                lastpart[0] = gsl_matrix_get(_initTEpos, sectidx, 0);
                lastpart[1] = gsl_matrix_get(_initTEpos, sectidx, 1);
                lastpart[2] = gsl_matrix_get(_initTEpos, sectidx, 2);
            }
            del_teposmag = sqrt(pow(tepos[0]-(lastpart[0]),2)
                              + pow(tepos[1]-(lastpart[1]),2)
                              + pow(tepos[2]-(lastpart[2]),2));

            vx = gsl_spline_eval(astvelx_spline, ista * (1. / spanres), accel_ptr);
            vy = gsl_spline_eval(astvely_spline, ista * (1. / spanres), accel_ptr);
            vz = gsl_spline_eval(astvelz_spline, ista * (1. / spanres), accel_ptr);
            vmag = sqrt(vx * vx + vy * vy + vz * vz);

            localflowdir[0] = vx / vmag;
            localflowdir[1] = vy / vmag;
            localflowdir[2] = vz / vmag;

            //assuming only a single peak in the circulation distribution for now
            //magnitude and sign of trail vortices
            if ((double) ista / spanres <= spandisc_peakcirc_lfn && ista != 0)
                trailvormag = -(gsl_spline_eval(circ_spline, ista * (1. / spanres), accel_ptr)
                                - gsl_spline_eval(circ_spline, (ista - 1) * (1. / spanres), accel_ptr));
            else if ((double) ista / spanres > spandisc_peakcirc_lfn && ista != spanres)
                trailvormag = gsl_spline_eval(circ_spline, ista * (1. / spanres), accel_ptr)
                              - gsl_spline_eval(circ_spline, (ista + 1) * (1. / spanres), accel_ptr);

            if (ista == 0)
                trailvormag = -circ_lfn[0];//if beginning (from root) of a new lfnline
                                           //clockwise vortex when looking from behind is -ve
            if (ista == spanres)
                trailvormag = circ_lfn[NbOfAst[ilfn] - 1];//last airstation gives anticlockwise vortex
                                                          //anticlockwise vortex is +ve

            if (suprootvor) {//suppressing effect of root vortices (avoid fountain effect in hover)
                if (ista < 0.15 * (spanres + 1)) {
                    //if (stepnum < transientsteps)
                    //    alphai = trailvormag * hres * stepnum / transientsteps;

                    //alphai = trailvormag*hres* ista/(0.2*(spanres + 1));
                    //alphai = trailvormag * hres * 0.1;
                } else
                    alphai = trailvormag * hres;
            } else
                alphai = trailvormag * hres;

            voli = 0.05 * hres;   //ip: not entirely sure about this (is this an empirical parameter??)
            //voli = (4.0*M_PI/3.0) * hres*hres*hres;
            num_part = (int) (del_teposmag / hres);  //number of particles behind each ista
            if (trailvor){
                for (int jsta = int(dontaddTEpart); jsta <= num_part; ++jsta) {
                    //evaluate trailing vortex parameters corresponding to a airsta
                    if (del_teposmag != 0.0) {//in case lfnline is moving in inertial frame
                        trailvorpos[0] = lastpart[0] + jsta * (tepos[0] - (lastpart[0])) / (del_teposmag / hres);
                        trailvorpos[1] = lastpart[1] + jsta * (tepos[1] - (lastpart[1])) / (del_teposmag / hres);
                        trailvorpos[2] = lastpart[2] + jsta * (tepos[2] - (lastpart[2])) / (del_teposmag / hres);
                    } else {//only stepnum=0 in case lfnline is stationary in inertial frame
                        //particles added at TE; thereafter del_teposmag !=0.0, particles added between last section particle and TE
                        trailvorpos[0] = lastpart[0];
                        trailvorpos[1] = lastpart[1];
                        trailvorpos[2] = lastpart[2];
                    }
                    for (int k = 0; k < 3; ++k) {
                        gsl_matrix_set(_position, _npidx, k, trailvorpos[k]);
                        gsl_matrix_set(_vorticity, _npidx, k, localflowdir[k] * alphai);
                        gsl_matrix_set(_vorticityfield, _npidx, k, 0.0); //later
                    }
                    gsl_vector_set(_radius, _npidx, sigma);
                    gsl_vector_set(_volume, _npidx, voli);
                    gsl_vector_set(_birthstrength, _npidx, fabs(alphai));
                    gsl_vector_set(_active, _npidx, stepnum);//unsused for now
                    gsl_vector_set(_lastsectpartnpidx, sectidx, _npidx);
                    if (_npidx == 0)
                        printf("First particle created at : %8.3e, %8.3e, %8.3e \n", trailvorpos[0], trailvorpos[1],
                               trailvorpos[2]);
                    _npidx++;
                    if (_npidx == MAXNUMPARTICLES) {
                        _npidx = 0;
                        _numParticles = MAXNUMPARTICLES;
                        _maxmem = true;
                    }
                    if (!_maxmem)
                        _numParticles = _npidx;
                }
            }
            if (shedvor) {
                if (ista != spanres) {//shed vortex only between airsta
                    sectidx = ilfn*(2*spanres+1)+2*ista+1;
                    teposnxt[0] = gsl_spline_eval(TEposx_spline, (ista+1) * (1. / spanres), accel_ptr);
                    teposnxt[1] = gsl_spline_eval(TEposy_spline, (ista+1) * (1. / spanres), accel_ptr);
                    teposnxt[2] = gsl_spline_eval(TEposz_spline, (ista+1) * (1. / spanres), accel_ptr);

                    teposprevnxt[0] = gsl_spline_eval(TEposxprev_spline, (ista+1) * (1. / spanres), accel_ptr);
                    teposprevnxt[1] = gsl_spline_eval(TEposyprev_spline, (ista+1) * (1. / spanres), accel_ptr);
                    teposprevnxt[2] = gsl_spline_eval(TEposzprev_spline, (ista+1) * (1. / spanres), accel_ptr);

                    for (int k = 0; k < 3; ++k) {
                        tepos_shed[k] = 0.5*(tepos[k] + teposnxt[k]);
                        teposprev_shed[k] = 0.5*(teposprev[k] + teposprevnxt[k]);
                    }

                    if (stepnum==0){
                        gsl_matrix_set(_initTEpos, sectidx, 0,teposprev_shed[0]);
                        gsl_matrix_set(_initTEpos, sectidx, 1,teposprev_shed[1]);
                        gsl_matrix_set(_initTEpos, sectidx, 2,teposprev_shed[2]);
                        dontaddTEpart = false;
                    }
                    if (gsl_vector_get(_lastsectpartnpidx, sectidx)!=DBL_MAX) {
                        lastpart_shed[0] = gsl_matrix_get(_position, gsl_vector_get(_lastsectpartnpidx, sectidx), 0);
                        lastpart_shed[1] = gsl_matrix_get(_position, gsl_vector_get(_lastsectpartnpidx, sectidx), 1);
                        lastpart_shed[2] = gsl_matrix_get(_position, gsl_vector_get(_lastsectpartnpidx, sectidx), 2);
                    }
                    else{
                        lastpart_shed[0] = gsl_matrix_get(_initTEpos, sectidx, 0);
                        lastpart_shed[1] = gsl_matrix_get(_initTEpos, sectidx, 1);
                        lastpart_shed[2] = gsl_matrix_get(_initTEpos, sectidx, 2);
                    }
                    del_teposmag = sqrt(  pow(tepos_shed[0]-(lastpart_shed[0]),2)
                                        + pow(tepos_shed[1]-(lastpart_shed[1]),2)
                                        + pow(tepos_shed[2]-(lastpart_shed[2]),2));
                    shedvordir[0] =   gsl_spline_eval(astposx_spline, ista * (1. / spanres), accel_ptr)
                                    - gsl_spline_eval(astposx_spline, (ista+1) * (1. / spanres), accel_ptr);
                    shedvordir[1] =   gsl_spline_eval(astposy_spline, ista * (1. / spanres), accel_ptr)
                                    - gsl_spline_eval(astposy_spline, (ista+1) * (1. / spanres), accel_ptr);
                    shedvordir[2] =   gsl_spline_eval(astposz_spline, ista * (1. / spanres), accel_ptr)
                                    - gsl_spline_eval(astposz_spline, (ista+1) * (1. / spanres), accel_ptr);

                    shedvordirmag = sqrt(shedvordir[0]*shedvordir[0] + shedvordir[1]*shedvordir[1] + shedvordir[2]*shedvordir[2]);

                    //ip: not exactly correct (since last station not reached?)
//                    shedvormag =   gsl_spline_eval(circ_spline,     ista * (1. / hres), accel_ptr)
//                                 - gsl_spline_eval(circprev_spline, ista * (1. / hres), accel_ptr);
                    shedvormag =   0.5*(   gsl_spline_eval(circ_spline,     ista *     (1. / spanres),     accel_ptr)
                                         + gsl_spline_eval(circ_spline,     (ista+1) * (1. / spanres),     accel_ptr))
                                 - 0.5*(   gsl_spline_eval(circprev_spline, ista *     (1. / spanres),     accel_ptr)
                                         + gsl_spline_eval(circprev_spline, (ista+1) * (1. / spanres),     accel_ptr));
                    //placing shed vortex particle between two trailing vortex particles
                    for (int ksta = int(dontaddTEpart); ksta <= (int) (del_teposmag / hres); ++ksta) {
                        if (del_teposmag != 0.0) {//in case lfnline is moving in inertial frame
                            shedvorpos[0] = lastpart_shed[0] + ksta * (tepos_shed[0] - lastpart_shed[0])/(del_teposmag/hres);
                            shedvorpos[1] = lastpart_shed[1] + ksta * (tepos_shed[1] - lastpart_shed[1])/(del_teposmag/hres);
                            shedvorpos[2] = lastpart_shed[2] + ksta * (tepos_shed[2] - lastpart_shed[2])/(del_teposmag/hres);

                        } else {//only stepnum=0 in case lfnline is stationary in inertial frame
                            //particles added at TE; thereafter del_teposmag !=0.0, particles added between last section particle and TE
                            shedvorpos[0] = lastpart_shed[0];
                            shedvorpos[1] = lastpart_shed[1];
                            shedvorpos[2] = lastpart_shed[2];

                        }
                        double alphai_shed = shedvormag * hres;
                        for (int k = 0; k < 3; ++k) {
                            gsl_matrix_set(_position, _npidx, k, shedvorpos[k]);
                            gsl_matrix_set(_vorticity, _npidx, k, shedvordir[k] * alphai_shed / shedvordirmag);
                            gsl_matrix_set(_vorticityfield, _npidx, k, 0.0); //later
                        }
                        gsl_vector_set(_radius, _npidx, sigma);
                        gsl_vector_set(_volume, _npidx, voli);
                        gsl_vector_set(_birthstrength, _npidx, fabs(alphai_shed));
                        gsl_vector_set(_active, _npidx, stepnum);//unsused for now
                        gsl_vector_set(_lastsectpartnpidx, sectidx, _npidx);

                        _npidx++;
                        if (_npidx == MAXNUMPARTICLES) {
                            _npidx = 0;
                            _numParticles = MAXNUMPARTICLES;
                            _maxmem = true;
                        }
                        if (!_maxmem)
                            _numParticles = _npidx;
                    }
                }
            }
        }
        gsl_spline_free(circ_spline);gsl_spline_free(circprev_spline);
        gsl_spline_free(TEposx_spline);gsl_spline_free(TEposy_spline);gsl_spline_free(TEposz_spline);
        gsl_spline_free(TEposxprev_spline);gsl_spline_free(TEposyprev_spline);gsl_spline_free(TEposzprev_spline);
        gsl_spline_free(astposx_spline);gsl_spline_free(astposy_spline);gsl_spline_free(astposz_spline);
        gsl_interp_accel_free(accel_ptr);
        free(TEposx_lfn);free(TEposy_lfn);free(TEposz_lfn);
        free(TEposxprev_lfn);free(TEposyprev_lfn);free(TEposzprev_lfn);
        free(circ_lfn);free(circprev_lfn);free(spandisc_lfn);
        free(astposx_lfn);free(astposy_lfn);free(astposz_lfn);
        free(astvelx_lfn);free(astvely_lfn);free(astvelz_lfn);
    }
    _size = 2*_numParticles*3;
}*/
void pawan::__wake::addParticles(PawanRecvData pawanrecvdata,size_t &stepnum){
    printf("\t adding particles..... \n");
    OUT("_numParticles",_numParticles);
    OUT("_npidx",_npidx);
    char *dymfilename = pawanrecvdata->Dymfilename;
    int ccr=0;  //whether it is a ccr case
    if(strstr(dymfilename,"ccr") != NULL)
        ccr=1;
    double *Vinf = pawanrecvdata->Vinf;
    int NbOfLfnLines = pawanrecvdata->NbOfLfnLines;
    int *NbOfAst = pawanrecvdata->NbOfAst;
    double *lfnlen = pawanrecvdata->lfnlen;
    int trailvor = pawanrecvdata->trailvor;
    int shedvor = pawanrecvdata->shedvor;
    int suprootvor = pawanrecvdata->suprootvor;
    if (suprootvor)
        printf("\t suppressing root vortex..... \n");
    int transientsteps = pawanrecvdata->transientsteps;
    int infl2D = pawanrecvdata->infl2D;
    int spanres = pawanrecvdata->spanres;//number of particles over the span of a lfnline
    //int spanres = 20;
    double acrossa = pawanrecvdata->acrossa;
    double deltat = pawanrecvdata->deltat;
    double *spandisc = pawanrecvdata->span_disc;
    double *TEpos = pawanrecvdata->TEpos;
    double *circ = pawanrecvdata->circ;
    double *TEpos_prev = pawanrecvdata->TEpos_prev;
    double *circ_prev = pawanrecvdata->circ_prev;
    double *astpos = pawanrecvdata->astpos;
    double *astvel = pawanrecvdata->astvel;

    double Vinfmag = sqrt(Vinf[0]*Vinf[0] + Vinf[1]*Vinf[1] + Vinf[2]*Vinf[2]);
    int astidx = 0;
    int offsetrows=0;
    int sectidx = 0; //index of section where particles originate (varies on lfnline from 0 to spanres; spanres+1 sections)
    int iast_peakcirc_lfn;
    double peakcirc_lfn, spandisc_peakcirc_lfn;
    double sigma, hres, alphai_trail, trailvormag;
    bool dontaddTEpart;
    double tepos[3],teposprev[3], lastpart[3]; // _, _, coordinates of last particle to emanate from a section ista
    double del_teposmag, vx, vy, vz, vmag, voli;
    double localflowdir[3], vorpos[3];  //need more astkin details to get localflowdir accurately for blades
    int num_part; //
    double shedvorpos[3],shedvordir[3],shedvormag,shedvordirmag, alphai_shed;
    double teposnxt[3], teposprevnxt[3], tepos_shed[3],teposprev_shed[3];
    double lastpart_shed[3]; //coordinates of last particle to emanate between ista and ista + 1
    double alphai, vorticity_net[3];
    for(size_t ilfn = 0; ilfn<NbOfLfnLines; ++ilfn) {

        double *TEposx_lfn = (double *) calloc(NbOfAst[ilfn], sizeof(double));
        double *TEposy_lfn = (double *) calloc(NbOfAst[ilfn], sizeof(double));
        double *TEposz_lfn = (double *) calloc(NbOfAst[ilfn], sizeof(double));
        getlfnvec(TEposx_lfn, TEpos, 3, 0, offsetrows, NbOfAst[ilfn]);
        getlfnvec(TEposy_lfn, TEpos, 3, 1, offsetrows, NbOfAst[ilfn]);
        getlfnvec(TEposz_lfn, TEpos, 3, 2, offsetrows, NbOfAst[ilfn]);
        double *TEposxprev_lfn = (double *) calloc(NbOfAst[ilfn], sizeof(double));
        double *TEposyprev_lfn = (double *) calloc(NbOfAst[ilfn], sizeof(double));
        double *TEposzprev_lfn = (double *) calloc(NbOfAst[ilfn], sizeof(double));
        getlfnvec(TEposxprev_lfn, TEpos_prev, 3, 0, offsetrows, NbOfAst[ilfn]);
        getlfnvec(TEposyprev_lfn, TEpos_prev, 3, 1, offsetrows, NbOfAst[ilfn]);
        getlfnvec(TEposzprev_lfn, TEpos_prev, 3, 2, offsetrows, NbOfAst[ilfn]);
        double *circ_lfn = (double *) calloc(NbOfAst[ilfn], sizeof(double));
        getlfnvec(circ_lfn, circ, 1, 0, offsetrows, NbOfAst[ilfn]);
        double *circprev_lfn = (double *) calloc(NbOfAst[ilfn], sizeof(double));
        getlfnvec(circprev_lfn, circ_prev, 1, 0, offsetrows, NbOfAst[ilfn]);
        double *astposx_lfn = (double *) calloc(NbOfAst[ilfn], sizeof(double));
        double *astposy_lfn = (double *) calloc(NbOfAst[ilfn], sizeof(double));
        double *astposz_lfn = (double *) calloc(NbOfAst[ilfn], sizeof(double));
        getlfnvec(astposx_lfn, astpos, 3, 0, offsetrows, NbOfAst[ilfn]);
        getlfnvec(astposy_lfn, astpos, 3, 1, offsetrows, NbOfAst[ilfn]);
        getlfnvec(astposz_lfn, astpos, 3, 2, offsetrows, NbOfAst[ilfn]);
        double *astvelx_lfn = (double *) calloc(NbOfAst[ilfn], sizeof(double));
        double *astvely_lfn = (double *) calloc(NbOfAst[ilfn], sizeof(double));
        double *astvelz_lfn = (double *) calloc(NbOfAst[ilfn], sizeof(double));
        getlfnvec(astvelx_lfn, astvel, 3, 0, offsetrows, NbOfAst[ilfn]);
        getlfnvec(astvely_lfn, astvel, 3, 1, offsetrows, NbOfAst[ilfn]);
        getlfnvec(astvelz_lfn, astvel, 3, 2, offsetrows, NbOfAst[ilfn]);
        double *spandisc_lfn = (double *) calloc(NbOfAst[ilfn], sizeof(double));
        getlfnvec(spandisc_lfn, spandisc, 1, 0, offsetrows, NbOfAst[ilfn]);
        offsetrows += NbOfAst[ilfn];//also just astidx can be used

        gsl_interp_accel *accel_ptr;
        gsl_spline *circ_spline, *TEposx_spline, *TEposy_spline, *TEposz_spline;
        gsl_spline *circprev_spline, *TEposxprev_spline, *TEposyprev_spline, *TEposzprev_spline;
        gsl_spline *astposx_spline, *astposy_spline, *astposz_spline;
        gsl_spline *astvelx_spline, *astvely_spline, *astvelz_spline;
        accel_ptr = gsl_interp_accel_alloc();
        circ_spline = gsl_spline_alloc(gsl_interp_cspline, NbOfAst[ilfn]);
        TEposx_spline = gsl_spline_alloc(gsl_interp_cspline, NbOfAst[ilfn]);
        TEposy_spline = gsl_spline_alloc(gsl_interp_cspline, NbOfAst[ilfn]);
        TEposz_spline = gsl_spline_alloc(gsl_interp_cspline, NbOfAst[ilfn]);
        circprev_spline = gsl_spline_alloc(gsl_interp_cspline, NbOfAst[ilfn]);
        TEposxprev_spline = gsl_spline_alloc(gsl_interp_cspline, NbOfAst[ilfn]);
        TEposyprev_spline = gsl_spline_alloc(gsl_interp_cspline, NbOfAst[ilfn]);
        TEposzprev_spline = gsl_spline_alloc(gsl_interp_cspline, NbOfAst[ilfn]);
        astposx_spline = gsl_spline_alloc(gsl_interp_cspline, NbOfAst[ilfn]);
        astposy_spline = gsl_spline_alloc(gsl_interp_cspline, NbOfAst[ilfn]);
        astposz_spline = gsl_spline_alloc(gsl_interp_cspline, NbOfAst[ilfn]);
        astvelx_spline = gsl_spline_alloc(gsl_interp_cspline, NbOfAst[ilfn]);
        astvely_spline = gsl_spline_alloc(gsl_interp_cspline, NbOfAst[ilfn]);
        astvelz_spline = gsl_spline_alloc(gsl_interp_cspline, NbOfAst[ilfn]);
        gsl_spline_init(circ_spline, spandisc_lfn, circ_lfn, NbOfAst[ilfn]);
        gsl_spline_init(TEposx_spline, spandisc_lfn, TEposx_lfn, NbOfAst[ilfn]);
        gsl_spline_init(TEposy_spline, spandisc_lfn, TEposy_lfn, NbOfAst[ilfn]);
        gsl_spline_init(TEposz_spline, spandisc_lfn, TEposz_lfn, NbOfAst[ilfn]);
        gsl_spline_init(circprev_spline, spandisc_lfn, circprev_lfn, NbOfAst[ilfn]);
        gsl_spline_init(TEposxprev_spline, spandisc_lfn, TEposxprev_lfn, NbOfAst[ilfn]);
        gsl_spline_init(TEposyprev_spline, spandisc_lfn, TEposyprev_lfn, NbOfAst[ilfn]);
        gsl_spline_init(TEposzprev_spline, spandisc_lfn, TEposzprev_lfn, NbOfAst[ilfn]);
        gsl_spline_init(astposx_spline, spandisc_lfn, astposx_lfn, NbOfAst[ilfn]);
        gsl_spline_init(astposy_spline, spandisc_lfn, astposy_lfn, NbOfAst[ilfn]);
        gsl_spline_init(astposz_spline, spandisc_lfn, astposz_lfn, NbOfAst[ilfn]);
        gsl_spline_init(astvelx_spline, spandisc_lfn, astvelx_lfn, NbOfAst[ilfn]);
        gsl_spline_init(astvely_spline, spandisc_lfn, astvely_lfn, NbOfAst[ilfn]);
        gsl_spline_init(astvelz_spline, spandisc_lfn, astvelz_lfn, NbOfAst[ilfn]);

        //get peak circulation and index
        iast_peakcirc_lfn = 0;          //airsta index where peak circulation occurs
        peakcirc_lfn = circ_lfn[0];  //peak lfnline circulation
        spandisc_peakcirc_lfn = 0.0; //non-dim radius where peak circ occurs
        for (int i = 1; i < NbOfAst[ilfn]; i++) {
            if (circ[i] > peakcirc_lfn) {
                peakcirc_lfn = circ_lfn[i];
                iast_peakcirc_lfn = i;
                spandisc_peakcirc_lfn = spandisc_lfn[i];
            }
        }
        printf("++++++++ Peak circ at : %8.3e ++++++ \n", spandisc_peakcirc_lfn);
        hres = lfnlen[ilfn] / spanres;
        sigma = 1.3 * hres;

        for (int ista = 0; ista < spanres + 1; ++ista) {//new discretized station index
            sectidx = ilfn * (spanres + 1) + ista;
            dontaddTEpart = true;  //whether to add particles right at the TE

            tepos[0] = gsl_spline_eval(TEposx_spline, ista * (1. / spanres), accel_ptr);
            tepos[1] = gsl_spline_eval(TEposy_spline, ista * (1. / spanres), accel_ptr);
            tepos[2] = gsl_spline_eval(TEposz_spline, ista * (1. / spanres), accel_ptr);

            teposprev[0] = gsl_spline_eval(TEposxprev_spline, ista * (1. / spanres), accel_ptr);
            teposprev[1] = gsl_spline_eval(TEposyprev_spline, ista * (1. / spanres), accel_ptr);
            teposprev[2] = gsl_spline_eval(TEposzprev_spline, ista * (1. / spanres), accel_ptr);

            if (stepnum == 0) {
                gsl_matrix_set(_initTEpos, sectidx, 0, teposprev[0]);
                gsl_matrix_set(_initTEpos, sectidx, 1, teposprev[1]);
                gsl_matrix_set(_initTEpos, sectidx, 2, teposprev[2]);
                dontaddTEpart = false; //add particle right at the TE only for the first time step
            }
            if (gsl_vector_get(_lastsectpartnpidx, sectidx)!=DBL_MAX) {
                lastpart[0] = gsl_matrix_get(_position, gsl_vector_get(_lastsectpartnpidx, sectidx), 0);
                lastpart[1] = gsl_matrix_get(_position, gsl_vector_get(_lastsectpartnpidx, sectidx), 1);
                lastpart[2] = gsl_matrix_get(_position, gsl_vector_get(_lastsectpartnpidx, sectidx), 2);
            }
            else{
                lastpart[0] = gsl_matrix_get(_initTEpos, sectidx, 0);
                lastpart[1] = gsl_matrix_get(_initTEpos, sectidx, 1);
                lastpart[2] = gsl_matrix_get(_initTEpos, sectidx, 2);
            }
            del_teposmag = sqrt(pow(tepos[0]-(lastpart[0]),2)
                              + pow(tepos[1]-(lastpart[1]),2)
                              + pow(tepos[2]-(lastpart[2]),2));

            vx = gsl_spline_eval(astvelx_spline, ista * (1. / spanres), accel_ptr);
            vy = gsl_spline_eval(astvely_spline, ista * (1. / spanres), accel_ptr);
            vz = gsl_spline_eval(astvelz_spline, ista * (1. / spanres), accel_ptr);
            vmag = sqrt(vx * vx + vy * vy + vz * vz);

            localflowdir[0] = vx / vmag;
            localflowdir[1] = vy / vmag;
            localflowdir[2] = vz / vmag;

            //assuming only a single peak in the circulation distribution for now
            //magnitude and sign of trail vortices
            trailvormag=0.0;
            if ((double) ista / spanres <= spandisc_peakcirc_lfn && ista != 0)
                trailvormag = -(gsl_spline_eval(circ_spline, ista * (1. / spanres), accel_ptr)
                                - gsl_spline_eval(circ_spline, (ista - 1) * (1. / spanres), accel_ptr));
            else if ((double) ista / spanres > spandisc_peakcirc_lfn && ista != spanres)
                trailvormag = gsl_spline_eval(circ_spline, ista * (1. / spanres), accel_ptr)
                              - gsl_spline_eval(circ_spline, (ista + 1) * (1. / spanres), accel_ptr);

            /*if (ista == 0)
                trailvormag = -circ_lfn[0];//if beginning (from root) of a new lfnline
            //clockwise vortex when looking from behind is -ve
            if (ista == spanres)
                trailvormag = circ_lfn[NbOfAst[ilfn] - 1];//last airstation gives anticlockwise vortex
            //anticlockwise vortex is +ve
*/
            if(ccr==1 && ilfn>1) //hard coded for the ccr case (for anticlockwise rotor vorticity direction is reversed)
                trailvormag = -trailvormag;

            alphai_trail = 0.0;
            if (suprootvor) {//suppressing effect of root vortices (avoid fountain effect in hover)
                if (ista < 0.5 * (spanres + 1)) {
                //if((float)(ista/(spanres + 1))<spandisc_peakcirc_lfn){
                    if (stepnum < transientsteps) {
                        alphai_trail = trailvormag * hres * stepnum / transientsteps;
                        //alphai_trail = trailvormag*hres* ista/(0.2*(spanres + 1));
                        //alphai_trail = trailvormag*hres*0.3;
                    }
                    else
                        //alphai_trail = trailvormag*hres*0.3;
                        alphai_trail = trailvormag * hres;
                } else
                    alphai_trail = trailvormag * hres;
            } else
                alphai_trail = trailvormag * hres;

            voli = hres * hres * hres;   //ip: not entirely sure about this (is this an empirical parameter??)
            for (int k = 0; k < 3; ++k)
                vorticity_net[k] = trailvor * localflowdir[k] * alphai_trail;
            if (shedvor) {
                if (ista != spanres) {//shed vortex only between airsta
                                        shedvordir[0] = gsl_spline_eval(astposx_spline, ista * (1. / spanres), accel_ptr)
                                                        - gsl_spline_eval(astposx_spline, (ista + 1) * (1. / spanres), accel_ptr);
                                        shedvordir[1] = gsl_spline_eval(astposy_spline, ista * (1. / spanres), accel_ptr)
                                                        - gsl_spline_eval(astposy_spline, (ista + 1) * (1. / spanres), accel_ptr);
                                        shedvordir[2] = gsl_spline_eval(astposz_spline, ista * (1. / spanres), accel_ptr)
                                                        - gsl_spline_eval(astposz_spline, (ista + 1) * (1. / spanres), accel_ptr);
                                        shedvordirmag = sqrt(shedvordir[0] * shedvordir[0] + shedvordir[1] * shedvordir[1] +
                                                             shedvordir[2] * shedvordir[2]);
                }
                //for the tip, the same wing TE direction is assumed
                shedvormag = gsl_spline_eval(circ_spline, ista * (1. / spanres), accel_ptr)
                             - gsl_spline_eval(circprev_spline, ista * (1. / spanres), accel_ptr);
                alphai_shed = shedvormag * hres;
                for (int k = 0; k < 3; ++k)
                    vorticity_net[k] += shedvor * shedvordir[k] * alphai_shed / shedvordirmag;
            }
            alphai = sqrt(vorticity_net[0]*vorticity_net[0] + vorticity_net[1]*vorticity_net[1] +vorticity_net[2]*vorticity_net[2]);

            num_part = (int) (del_teposmag / hres);  //number of particles behind each ista
            for (int jsta = int(dontaddTEpart); jsta <= num_part; ++jsta) {
                //evaluate trailing vortex parameters corresponding to a airsta
                if (del_teposmag != 0.0) {//in case lfnline is moving in inertial frame
                    vorpos[0] = lastpart[0] + jsta * (tepos[0] - (lastpart[0])) / (del_teposmag / hres);
                    vorpos[1] = lastpart[1] + jsta * (tepos[1] - (lastpart[1])) / (del_teposmag / hres);
                    vorpos[2] = lastpart[2] + jsta * (tepos[2] - (lastpart[2])) / (del_teposmag / hres);
                } else {//only stepnum=0 in case lfnline is stationary in inertial frame
                    //particles added at TE; thereafter del_teposmag !=0.0, particles added between last section particle and TE
                    vorpos[0] = lastpart[0];
                    vorpos[1] = lastpart[1];
                    vorpos[2] = lastpart[2];
                }
                for (int k = 0; k < 3; ++k) {
                    gsl_matrix_set(_position, _npidx, k, vorpos[k]);
                    gsl_matrix_set(_vorticity, _npidx, k, vorticity_net[k]);
                    gsl_matrix_set(_vorticityfield, _npidx, k, 0.0); //later
                }
                gsl_vector_set(_radius, _npidx, sigma);
                gsl_vector_set(_volume, _npidx, voli);
                gsl_vector_set(_birthstrength, _npidx, alphai);
                gsl_vector_set(_active, _npidx, stepnum);
                gsl_vector_set(_lastsectpartnpidx, sectidx, _npidx);
                if (_npidx == 0)
                    printf("First particle created at : %8.3e, %8.3e, %8.3e \n", vorpos[0], vorpos[1],
                           vorpos[2]);
                _npidx++;
                if (_npidx == MAXNUMPARTICLES) {
                    _npidx = 0;
                    _numParticles = MAXNUMPARTICLES;
                    _maxmem = true;
                }
                if (!_maxmem)
                    _numParticles = _npidx;
            }
        }
        gsl_spline_free(circ_spline);gsl_spline_free(circprev_spline);
        gsl_spline_free(TEposx_spline);gsl_spline_free(TEposy_spline);gsl_spline_free(TEposz_spline);
        gsl_spline_free(TEposxprev_spline);gsl_spline_free(TEposyprev_spline);gsl_spline_free(TEposzprev_spline);
        gsl_spline_free(astposx_spline);gsl_spline_free(astposy_spline);gsl_spline_free(astposz_spline);
        gsl_interp_accel_free(accel_ptr);
        free(TEposx_lfn);free(TEposy_lfn);free(TEposz_lfn);
        free(TEposxprev_lfn);free(TEposyprev_lfn);free(TEposzprev_lfn);
        free(circ_lfn);free(circprev_lfn);free(spandisc_lfn);
        free(astposx_lfn);free(astposy_lfn);free(astposz_lfn);
        free(astvelx_lfn);free(astvely_lfn);free(astvelz_lfn);
    }
    _size = 2*_numParticles*3;
}
pawan::__wake::__wake(PawanRecvData pawanrecvdata){

    initialise_memory();   //large enough memory allocation
    _lastsectpartnpidx = gsl_vector_calloc(PAWAN_MAXLFNLINES*400);   //expecting a maximum spanres of 399
    gsl_vector_set_all(_lastsectpartnpidx,DBL_MAX);
    _initTEpos = gsl_matrix_calloc(PAWAN_MAXLFNLINES*400,3);
    size_t stepnum=0;
    addParticles(pawanrecvdata, stepnum);
}

void pawan::__wake::addParticles(__wake *W){

    for(int i = 0; i<W->_numParticles; ++i){
        for(int j = 0; j<3; ++j){
            gsl_matrix_set(_position,i,j,gsl_matrix_get(W->_position,i,j));
            gsl_matrix_set(_vorticity,i,j,gsl_matrix_get(W->_vorticity,i,j));
            gsl_matrix_set(_vorticityfield,i,j,gsl_matrix_get(W->_vorticityfield,i,j));
        }
        gsl_vector_set(_radius,i,gsl_vector_get(W->_radius,i));
        gsl_vector_set(_volume,i,gsl_vector_get(W->_volume,i));
        gsl_vector_set(_birthstrength,i,gsl_vector_get(W->_birthstrength,i));
    }
    _numParticles = W->_numParticles;
    _size = 2*_numParticles*3;
}

pawan::__wake::__wake(__wake *W){

    initialise_memory();   //large enough memory allocation
    addParticles(W);
}

pawan::__wake::__wake(const int &n,FILE *f){

    create_particles(n);
    read(f);               //read wake info from file
    _size = 2*_numParticles*3;
}

void pawan::__wake::updateVinfEffect(const double *Vinf, double &dt){

    for(int i = 0; i<_numParticles; ++i) {
        for (int j = 0; j < 3; ++j) {
            if(i==0 && j==0)
                printf("Position of 1st particle     before update: %+8.3e, %+8.3e, %+8.3e\n",
                       gsl_matrix_get(_position, 0, 0),gsl_matrix_get(_position, 0, 1),gsl_matrix_get(_position, 0, 2));
            gsl_matrix_set(_position, i, j, Vinf[j]*dt + gsl_matrix_get(_position, i, j));
            if(i==0 && j==0)
                printf("Position of 1st particle     after update: %+8.3e, %+8.3e, %+8.3e\n",
                       gsl_matrix_get(_position, 0, 0),gsl_matrix_get(_position, 0, 1),gsl_matrix_get(_position, 0, 2));
        }
    }
}

void pawan::__wake::updateBoundVorBoundVorEffectVind(PawanRecvData pawanrecvdata, PawanSendData pawansenddata, int astidx0, int lfnidx0){
    //printf("incl bound vor - bound vor effect ----------------  \n");
    //printf("astidx0 = %d, lfnidx0= %d\n",astidx0,lfnidx0);
    int NbOfLfnLines = pawanrecvdata->NbOfLfnLines;
    int *NbOfAst = pawanrecvdata->NbOfAst;
    double *astpos = pawanrecvdata->astpos;
    double *circ = pawanrecvdata->circ;

    double *lambda = pawansenddata->lambda;

    gsl_vector *rast = gsl_vector_calloc(3);//airstation where ind vel is evaluated
    gsl_vector *vb = gsl_vector_calloc(3);//ind vel due to all panels
    gsl_vector *Si = gsl_vector_calloc(3);
    gsl_vector *rast1 = gsl_vector_calloc(3);
    gsl_vector *rast2 = gsl_vector_calloc(3);
    gsl_vector *rast12 = gsl_vector_calloc(3);
    gsl_vector *rast1r = gsl_vector_calloc(3);
    gsl_vector *licap = gsl_vector_calloc(3);
    gsl_vector *sicap = gsl_vector_calloc(3);
    gsl_vector *rast2r = gsl_vector_calloc(3);
    gsl_vector *vbi = gsl_vector_calloc(3);  //ind vel due to one panel
    double t,span,simag,alpha,beta;
    double circ1, circ2, gamma0, gamma1;
    double factor;

    gsl_vector_set_zero(vb);
    for (int i = 0; i < 3; ++i) {
        gsl_vector_set(rast, i, astpos[3 * astidx0 + i]);
    }

    int astidx_low=0;
    for (size_t ilfn = 0; ilfn < lfnidx0; ++ilfn) {
        astidx_low += NbOfAst[ilfn];
    }
    int astidx_up = astidx_low+NbOfAst[lfnidx0];
    int astidx=0;

    //printf("astidx_low = %d, astidx_up= %d\n",astidx_low,astidx_up);
    for (size_t ilfn = 0; ilfn < NbOfLfnLines; ++ilfn) {
        for (size_t iast = 0; iast < NbOfAst[ilfn] - 1; ++iast) {
            if(astidx < astidx_low || astidx >= astidx_up ) { //ignore airstations on same lfnline
                for (int j = 0; j < 3; ++j) {
                    gsl_vector_set(rast1, j, astpos[3 * astidx + j]);
                    gsl_vector_set(rast2, j, astpos[3 * (astidx + 1) + j]);
                }

                gsl_vector_memcpy(rast12, rast2);
                gsl_vector_sub(rast12, rast1);
                span = gsl_blas_dnrm2(rast12); //blade element span
                gsl_vector_memcpy(rast1r, rast);
                gsl_vector_sub(rast1r, rast1);
                gsl_blas_ddot(rast1r, rast12, &t);
                t = t / (span * span);

                gsl_vector_memcpy(Si, rast12);
                gsl_blas_dscal(t, Si);
                gsl_vector_add(Si, rast1);

                gsl_vector_memcpy(licap, rast12);
                gsl_blas_dscal(1 / span, licap);

                gsl_vector_memcpy(sicap, rast);
                gsl_vector_sub(sicap, Si);
                simag = gsl_blas_dnrm2(sicap);
                gsl_blas_dscal(1 / simag, sicap);

                gsl_vector_memcpy(rast2r, rast);
                gsl_vector_sub(rast2r, rast2);
                //double alpha = 0.0;
                gsl_blas_ddot(rast2r, rast12, &alpha);
                alpha = acos(-alpha / gsl_blas_dnrm2(rast2r) / span);
                //double beta = 0.0;
                gsl_blas_ddot(rast1r, rast12, &beta);
                beta = acos(beta / gsl_blas_dnrm2(rast1r) / span);

                circ1 = circ[astidx];
                circ2 = circ[astidx + 1];
                //t can be -ve
                gamma0 = (circ1 * fabs(span * (1 - t)) + circ2 * fabs(t * span)) / (span * span);
                gamma1 = (circ2 - circ1) / span;
                gsl_cross(licap, sicap, vbi);
                gsl_blas_dscal(0.25 * M_1_PI / simag, vbi);
                factor = 0.5 * span * gamma0 * (cos(beta) + cos(alpha)) + simag * gamma1 * (sin(beta) - sin(alpha));
                gsl_blas_dscal(factor, vbi);

                gsl_vector_add(vb, vbi);
                //printf("%4d,",astidx);
            }
            astidx++;
        }
        astidx++; //skip the ast at the end of the lfn
    }
    //printf("\n");
    //printf("lambda initial = %+10.5e, %+10.5e, %+10.5e \n",
    //       lambda[astidx0 * 3], lambda[astidx0 * 3 + 1], lambda[astidx0 * 3 + 2]);
    /*double time = pawanrecvdata->t;
    double num;
    double denom;
    double mean;
    double ct;
    double infl_zone = 45.0;
    double delta_time = (1.0/15.0)*(0.5*infl_zone/360.0);
    double gaussian=0.0;
    for(size_t rev = 0; rev < 15; ++rev) {
        for(size_t pass=0; pass<4; ++pass){
            ct = 0.1 + rev*0.066666 + pass*0.25*0.066666;
            if (time >= ct - delta_time && time < ct + delta_time) {
                mean = 0.1 + rev * 0.066666 + pass*0.25*0.066666;
                num = -(time - mean)*(time - mean);
                denom = 2*15*15*(360/infl_zone)*(360/infl_zone)*3*3*2*2; //30 degree zone of influence about blade passage
                gaussian=exp(num/denom);
                rev = 15;
                pass = 4;
                break;
            }
        }
    }
    gsl_blas_dscal(gaussian,vb);*/
    for (size_t k = 0; k < 3; ++k) {
        lambda[astidx0*3 + k] += gsl_vector_get(vb, k);
    }
    //printf("delta lambda = %+10.5e, %+10.5e, %+10.5e \n",
    //       gsl_vector_get(vb, 0), gsl_vector_get(vb, 1), gsl_vector_get(vb, 2));

    gsl_vector_free(vb);
    gsl_vector_free(Si);
    gsl_vector_free(rast1);
    gsl_vector_free(rast2);
    gsl_vector_free(rast12);
    gsl_vector_free(rast1r);
    gsl_vector_free(licap);
    gsl_vector_free(sicap);
    gsl_vector_free(rast2r);
    gsl_vector_free(vbi);
}

void pawan::__wake::BoundVorEffectVind(PawanRecvData pawanrecvdata,double &dt,size_t &stepnum){
    int bndvorinclnsteps = pawanrecvdata->bndvorinclnsteps;
    printf("incl bound vor vind effect ---------------- upto %d steps \n", bndvorinclnsteps);
    int NbOfLfnLines = pawanrecvdata->NbOfLfnLines;
    int *NbOfAst = pawanrecvdata->NbOfAst;
    double *astpos = pawanrecvdata->astpos;
    double *circ = pawanrecvdata->circ;

    gsl_vector *vb = gsl_vector_calloc(3);
    gsl_vector *Si = gsl_vector_calloc(3);
    gsl_vector *rast1 = gsl_vector_calloc(3);
    gsl_vector *rast2 = gsl_vector_calloc(3);
    gsl_vector *rast12 = gsl_vector_calloc(3);
    gsl_vector *rast1r = gsl_vector_calloc(3);
    gsl_vector *licap = gsl_vector_calloc(3);
    gsl_vector *sicap = gsl_vector_calloc(3);
    gsl_vector *rast2r = gsl_vector_calloc(3);
    gsl_vector *vbi = gsl_vector_calloc(3);
    double t,span,simag,alpha,beta;
    double circ1, circ2, gamma0, gamma1;
    double factor;
    for(int i = 0; i<_numParticles; ++i) {
        if(stepnum - gsl_vector_get(_active,i)<bndvorinclnsteps){ //only particles from the last ... steps affected by bound vortices
            gsl_vector_set_zero(vb);
            gsl_vector_const_view r = gsl_matrix_const_row(_position, i);
            int astidx = 0;
            for (size_t ilfn = 0; ilfn < NbOfLfnLines; ++ilfn) {
                for (size_t iast = 0; iast < NbOfAst[ilfn] - 2; ++iast) {
                    for (int j = 0; j < 3; ++j) {
                        gsl_vector_set(rast1, j, astpos[3 * astidx + j]);
                        gsl_vector_set(rast2, j, astpos[3 * (astidx + 1) + j]);
                    }

                    gsl_vector_memcpy(rast12, rast2);
                    gsl_vector_sub(rast12, rast1);
                    span = gsl_blas_dnrm2(rast12); //blade element span
                    gsl_vector_memcpy(rast1r, &r.vector);
                    gsl_vector_sub(rast1r, rast1);
                    gsl_blas_ddot(rast1r, rast12, &t);
                    t = t / (span * span);

                    gsl_vector_memcpy(Si, rast12);
                    gsl_blas_dscal(t, Si);
                    gsl_vector_add(Si, rast1);

                    gsl_vector_memcpy(licap, rast12);
                    gsl_blas_dscal(1/span, licap);

                    gsl_vector_memcpy(sicap, &r.vector);
                    gsl_vector_sub(sicap, Si);
                    simag = gsl_blas_dnrm2(sicap);
                    gsl_blas_dscal(1/simag, sicap);

                    gsl_vector_memcpy(rast2r, &r.vector);
                    gsl_vector_sub(rast2r, rast2);
                    //double alpha = 0.0;
                    gsl_blas_ddot(rast2r, rast12, &alpha);
                    alpha = acos(-alpha / gsl_blas_dnrm2(rast2r) / span);
                    //double beta = 0.0;
                    gsl_blas_ddot(rast1r, rast12, &beta);
                    beta = acos(beta / gsl_blas_dnrm2(rast1r) / span);

                    circ1 = circ[astidx];
                    circ2 = circ[astidx + 1];
                    //t can be -ve
                    gamma0 = (circ1 * fabs(span * (1 - t)) + circ2 * fabs(t * span)) / (span * span);
                    gamma1 = (circ2 - circ1) / span;
                    gsl_cross(licap, sicap, vbi);
                    gsl_blas_dscal(0.25 * M_1_PI / simag, vbi);
                    factor = span * gamma0 * (cos(beta) + cos(alpha)) + simag * gamma1 * (sin(beta) - sin(alpha));
                    gsl_blas_dscal(factor, vbi);

                    gsl_vector_add(vb, vbi);

                    astidx++;
                }
            }
            for (int j = 0; j < 3; ++j) {
                gsl_matrix_set(_position, i, j,
                               gsl_vector_get(vb, j) * dt + gsl_matrix_get(_position, i, j));
            }
        }
    }
    gsl_vector_free(vb);
    gsl_vector_free(Si);
    gsl_vector_free(rast1);
    gsl_vector_free(rast2);
    gsl_vector_free(rast12);
    gsl_vector_free(rast1r);
    gsl_vector_free(licap);
    gsl_vector_free(sicap);
    gsl_vector_free(rast2r);
    gsl_vector_free(vbi);
}

void pawan::__wake::BoundVorEffectStretch(PawanRecvData pawanrecvdata,double &dt,size_t &stepnum){
    int bndvorinclnsteps = pawanrecvdata->bndvorinclnsteps;
    printf("incl bound vor stretch effect ---------------- upto %d steps \n", bndvorinclnsteps);
    int NbOfLfnLines = pawanrecvdata->NbOfLfnLines;
    int *NbOfAst = pawanrecvdata->NbOfAst;
    double *astpos = pawanrecvdata->astpos;
    double *circ = pawanrecvdata->circ;

    gsl_vector *da = gsl_vector_calloc(3);
    gsl_vector *Si = gsl_vector_calloc(3);
    gsl_vector *rast1 = gsl_vector_calloc(3);
    gsl_vector *rast2 = gsl_vector_calloc(3);
    gsl_vector *rast12 = gsl_vector_calloc(3);
    gsl_vector *rast1r = gsl_vector_calloc(3);
    gsl_vector *licap = gsl_vector_calloc(3);
    gsl_vector *sicap = gsl_vector_calloc(3);
    gsl_vector *rast2r = gsl_vector_calloc(3);
    gsl_vector *d1cap = gsl_vector_calloc(3);
    gsl_vector *d2cap = gsl_vector_calloc(3);
    gsl_vector *da1 = gsl_vector_calloc(3);
    gsl_vector *licrosssi = gsl_vector_calloc(3);
    gsl_vector *da21 = gsl_vector_calloc(3);
    gsl_vector *da22 = gsl_vector_calloc(3);
    gsl_vector *da23 = gsl_vector_calloc(3);
    gsl_vector *da24 = gsl_vector_calloc(3);
    double t,span,simag,alpha,beta;
    double circ1, circ2, gamma0, gamma1;
    double factor1, factor21, factor22, factor23, factor24;
    double d1mag, d2mag, dotprod;
    for(int i = 0; i<_numParticles; ++i) {
        if (stepnum - gsl_vector_get(_active, i) < bndvorinclnsteps) { //only particles from the last ... steps affected by bound vortices
            gsl_vector_set_zero(da);
            gsl_vector_const_view a = gsl_matrix_const_row(_vorticity, i);
            gsl_vector_const_view r = gsl_matrix_const_row(_position, i);
            int astidx = 0;
            for (size_t ilfn = 0; ilfn < NbOfLfnLines; ++ilfn) {
                for (size_t iast = 0; iast < NbOfAst[ilfn] - 2; ++iast) {
                    for (int j = 0; j < 3; ++j) {
                        gsl_vector_set(rast1, j, astpos[3 * astidx + j]);
                        gsl_vector_set(rast2, j, astpos[3 * (astidx + 1) + j]);
                    }

                    gsl_vector_memcpy(rast12, rast2);
                    gsl_vector_sub(rast12, rast1);
                    span = gsl_blas_dnrm2(rast12); //blade element span
                    gsl_vector_memcpy(rast1r, &r.vector);
                    gsl_vector_sub(rast1r, rast1);
                    gsl_blas_ddot(rast1r, rast12, &t);
                    t = t / (span * span);

                    gsl_vector_memcpy(Si, rast12);
                    gsl_blas_dscal(t, Si);
                    gsl_vector_add(Si, rast1);

                    gsl_vector_memcpy(licap, rast12);
                    gsl_blas_dscal(1/span, licap);

                    gsl_vector_memcpy(sicap, &r.vector);
                    gsl_vector_sub(sicap, Si);
                    simag = gsl_blas_dnrm2(sicap);
                    gsl_blas_dscal(1/simag, sicap);

                    gsl_vector_memcpy(rast2r, &r.vector);
                    gsl_vector_sub(rast2r, rast2);
                    //double alpha = 0.0;
                    gsl_blas_ddot(rast2r, rast12, &alpha);
                    alpha = acos(-alpha / gsl_blas_dnrm2(rast2r) / span);
                    //double beta = 0.0;
                    gsl_blas_ddot(rast1r, rast12, &beta);
                    beta = acos(beta / gsl_blas_dnrm2(rast1r) / span);

                    circ1 = circ[astidx];
                    circ2 = circ[astidx + 1];
                    //t can be -ve
                    gamma0 = (circ1 * fabs(span * (1 - t)) + circ2 * fabs(t * span)) / (span * span);
                    gamma1 = (circ2 - circ1) / span;

                    gsl_vector_memcpy(d2cap, rast2r);
                    gsl_vector_memcpy(d1cap, rast1r);
                    d2mag = gsl_blas_dnrm2(rast2r);
                    d1mag = gsl_blas_dnrm2(rast1r);
                    gsl_blas_dscal(1/d2mag, d2cap);
                    gsl_blas_dscal(1/d1mag, d1cap);

                    gsl_cross(licap, &a.vector, da1);
                    factor1 = span * gamma0 * (cos(beta) + cos(alpha)) + simag * gamma1 * (sin(beta) - sin(alpha));
                    gsl_blas_dscal(factor1, da1);

                    gsl_cross(licap, sicap, licrosssi);
                    gsl_blas_ddot(&a.vector, licrosssi, &dotprod);

                    gsl_vector_memcpy(da21, licap);
                    factor21 = span * gamma0 * (sin(beta) - sin(alpha)) - simag * gamma1 * (cos(beta) + cos(alpha));
                    gsl_blas_dscal(factor21, da21);
                    gsl_vector_memcpy(da22, sicap);
                    factor22 = -(2 * span * gamma0 * (cos(beta) + cos(alpha)) -
                                 simag * gamma1 * (sin(beta) - sin(alpha)));
                    gsl_blas_dscal(factor22, da22);
                    gsl_vector_memcpy(da23, d1cap);
                    factor23 = -(span * gamma0 * sin(beta) - simag * gamma1 * cos(beta)) * cos(beta);
                    gsl_blas_dscal(factor23, da23);
                    gsl_vector_memcpy(da24, d2cap);
                    factor24 = -(span * gamma0 * sin(alpha) + simag * gamma1 * cos(alpha)) * cos(alpha);
                    gsl_blas_dscal(factor24, da24);

                    gsl_vector_add(da21, da22);
                    gsl_vector_add(da21, da23);
                    gsl_vector_add(da21, da24);
                    gsl_blas_dscal(dotprod, da21);
                    gsl_vector_add(da1, da21);
                    gsl_blas_dscal(0.25 * M_1_PI / (simag * simag), da1);
                    gsl_vector_add(da, da1);

                    astidx++;
                }
            }
            //printf("da1 =%+10.5e, %+10.5e, %+10.5e ---",
            //       gsl_vector_get(da1, 0), gsl_vector_get(da1, 1), gsl_vector_get(da1, 2));

            for (int j = 0; j < 3; ++j) {
                gsl_matrix_set(_vorticity, i, j,
                               gsl_vector_get(da, j) * dt + gsl_matrix_get(_vorticity, i, j));
            }
        }
    }
    gsl_vector_free(da);
    gsl_vector_free(Si);
    gsl_vector_free(rast1);
    gsl_vector_free(rast2);
    gsl_vector_free(rast12);
    gsl_vector_free(rast1r);
    gsl_vector_free(licap);
    gsl_vector_free(sicap);
    gsl_vector_free(rast2r);
    gsl_vector_free(d1cap);
    gsl_vector_free(d2cap);
    gsl_vector_free(da1);
    gsl_vector_free(licrosssi);
    gsl_vector_free(da21);
    gsl_vector_free(da22);
    gsl_vector_free(da23);
    gsl_vector_free(da24);

}

void pawan::__wake::updateBoundVorEffect(PawanRecvData pawanrecvdata,double &dt,size_t &stepnum){
    BoundVorEffectVind(pawanrecvdata,dt,stepnum);
    BoundVorEffectStretch(pawanrecvdata,dt,stepnum);
}

pawan::__wake::~__wake(){
    DOUT("------------------------------------------------------in pawan::__wake::~__wake()");
	gsl_matrix_free(_position);
	gsl_matrix_free(_velocity);
	gsl_matrix_free(_vorticity);
	gsl_matrix_free(_retvorcity);
	gsl_vector_free(_radius);
	gsl_vector_free(_volume);
	gsl_vector_free(_birthstrength);
	gsl_matrix_free(_vorticityfield);
    gsl_vector_free(_active);
}

void pawan::__wake::print(){
    DOUT("------------------------------------------------------in pawan::__wake::print()");
	OUT("_numParticles",_numParticles);
	OUT("_position",_position);
	OUT("_vorticity",_vorticity);
	OUT("_radius",_radius);
	OUT("_volume",_volume);
	OUT("_birthstrength",_birthstrength);
	OUT("_vorticityfield",_vorticityfield);
    OUT("_active",_active);
}

void pawan::__wake::save(FILE *f){
    DOUT("------------------------------------------------------in pawan::__wake::save()");
	fwrite(&_numParticles,sizeof(size_t),1,f);
	gsl_matrix_fwrite(f,_position);
	gsl_matrix_fwrite(f,_vorticity);
	gsl_vector_fwrite(f,_radius);
	gsl_vector_fwrite(f,_volume);
	gsl_vector_fwrite(f,_birthstrength);
	//gsl_matrix_fwrite(f,_vorticityfield);
}

void pawan::__wake::write(FILE *f){
    DOUT("------------------------------------------------------in pawan::__wake::write()");
	fwrite(&_numParticles,sizeof(size_t),1,f);
	gsl_matrix_fwrite(f,_position);
	gsl_matrix_fwrite(f,_vorticity);
	gsl_vector_fwrite(f,_radius);
    gsl_vector_fwrite(f,_active);
    gsl_vector_fwrite(f,_volume);
    gsl_vector_fwrite(f,_birthstrength);
}

void pawan::__wake::read(__io *IO){
    DOUT("------------------------------------------------------in pawan::__wake::read()");
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

void pawan::__wake::read(FILE *f){
    fread(&_numParticles,sizeof(size_t),1,f);
    gsl_matrix_fread(f,_position);
    gsl_matrix_fread(f,_vorticity);
    gsl_vector_fread(f,_radius);
    gsl_vector_fread(f,_active);
    gsl_vector_fread(f,_volume);
    gsl_vector_fread(f,_birthstrength);
    fclose(f);
}

void pawan::__wake::setStates(const gsl_vector *state){
    DOUT("------------------------------------------------------in pawan::__wake::setStates()");
	//OUT("setStates");
	//size_t np = state->size/2/3;
	size_t offset = _maxnumParticles*3;
	size_t vecsize = _maxnumParticles*3;
	gsl_matrix_const_view pos = gsl_matrix_const_view_vector(state,_maxnumParticles,3);
	gsl_matrix_memcpy(_position,&pos.matrix);
	gsl_vector_const_view vor = gsl_vector_const_subvector(state,offset,vecsize);
	gsl_matrix_const_view vrx = gsl_matrix_const_view_vector(&vor.vector,_maxnumParticles,3);
	gsl_matrix_memcpy(_vorticity,&vrx.matrix);
}

void pawan::__wake::getRates(gsl_vector *rate){
    DOUT("------------------------------------------------------in pawan::__wake::getRates()");
	//OUT("getRates");
	for(size_t i = 0; i<_numParticles; ++i){
		for(size_t j = 0; j<3; ++j){
			size_t ind = i*3 + j;
			gsl_vector_set(rate,ind,gsl_matrix_get(_velocity,i,j));
			ind += _maxsize/2;
			gsl_vector_set(rate,ind,gsl_matrix_get(_retvorcity,i,j));
		}
	}
}

void pawan::__wake::getStates(gsl_vector *state){
    DOUT("------------------------------------------------------in pawan::__wake::getStates()");
	//OUT("getStates");
	for(size_t i = 0; i<_numParticles; ++i){
		for(size_t j = 0; j<3; ++j){
			size_t ind = i*3 + j;
			gsl_vector_set(state,ind,gsl_matrix_get(_position,i,j));
			ind += _maxsize/2;
			gsl_vector_set(state,ind,gsl_matrix_get(_vorticity,i,j));
		}
	}
}

void pawan::__wake::getIdealRates(gsl_vector *rate){
    DOUT("------------------------------------------------------in pawan::__wake::getIdealRates()");
	//OUT("getIdealRates");
	gsl_vector_set_zero(rate);
}

void pawan::__wake::translate(const size_t &n, const double &x){
    DOUT("------------------------------------------------------in pawan::__wake::translate()");
	for(int i = 0; i<_numParticles; ++i){
		gsl_matrix_set(_position,i,n,x + gsl_matrix_get(_position,i,n));
	}
}

void pawan::__wake::translate(const double *x){
    for(size_t i = 0; i<_numParticles; ++i){
        for(size_t j = 0; j<3; ++j) {
            gsl_matrix_set(_position, i, j, x[j] + gsl_matrix_get(_position, i, j));
        }
    }
}

void pawan::__wake::rotate(const size_t &n,const double &a){
    double x,y,z;
    double xv,yv,zv;
    for(size_t i = 0; i<_numParticles; ++i){
        x = gsl_matrix_get(_position, i, 0);
        y = gsl_matrix_get(_position, i, 1);
        z = gsl_matrix_get(_position, i, 2);
        xv = gsl_matrix_get(_vorticity, i, 0);
        yv = gsl_matrix_get(_vorticity, i, 1);
        zv = gsl_matrix_get(_vorticity, i, 2);

        if (n==0) { //rotation about x-axis
            gsl_matrix_set(_position, i, 1,
                           cos(a) * y - sin(a) * z);
            gsl_matrix_set(_position, i, 2,
                           sin(a) * y + cos(a) * z);
            gsl_matrix_set(_vorticity, i, 1,
                           cos(a) * yv - sin(a) * zv);
            gsl_matrix_set(_vorticity, i, 2,
                           sin(a) * yv + cos(a) * zv);
        }
        else if (n==1) { //rotation about y-axis
            gsl_matrix_set(_position, i, 0,
                           cos(a) * x + sin(a) * z);
            gsl_matrix_set(_position, i, 2,
                           -sin(a) * x + cos(a) * z);
            gsl_matrix_set(_vorticity, i, 0,
                           cos(a) * xv + sin(a) * zv);
            gsl_matrix_set(_vorticity, i, 2,
                           -sin(a) * xv + cos(a) * zv);
        }
        else if (n==2) { //rotation about z-axis
            gsl_matrix_set(_position, i, 0,
                           cos(a) * x - sin(a) * y);
            gsl_matrix_set(_position, i, 1,
                           sin(a) * x + cos(a) * y);
            gsl_matrix_set(_vorticity, i, 0,
                           cos(a) * xv - sin(a) * yv);
            gsl_matrix_set(_vorticity, i, 1,
                           sin(a) * xv + cos(a) * yv);
        }
    }

}