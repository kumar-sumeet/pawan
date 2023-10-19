/*! PArticle Wake ANalysis
 * \file integration.cpp
 * \brief Routines for Integrations
 *
 * @author Puneet Singh
 * @date 04/21/2021
 */
#include "integration.h"
#include <unistd.h>

pawan::__integration::__integration(const double &t, const size_t &n){
    DOUT("-----------------------in pawan::__integration::__integration()");
	_dt = t/n;
	_t = t;
	_n = n;
}

pawan::__integration::__integration(){
    _dt = 0.0;
    _t = 0.0;
    _n = 0;
}

void pawan::__integration::integrate(__system *S,
                                     __io *IO,
                                     bool diagnose) {
    gsl_vector *states = gsl_vector_calloc(S->_totalmaxsize); //ip: have the *rates pointer here outside too for speed up
    FILE *f = IO->create_binary_file(".wake");
    FILE *fdiag = IO->create_binary_file(".diagnosis");
    double t = 0.0;
    fwrite(&t,sizeof(double),1,f);
    S->write(f);  //write particles info as is
    if(diagnose) {
        S->writenu(fdiag);
        fwrite(&t,sizeof(double),1,fdiag);
        S->diagnose();
        S->writediagnosis(fdiag);
    }
    double tStart = TIME();
    for(size_t i = 1; i<=_n; ++i){
        S->getStates(states);
        OUT("\tStep",i);
        t = i*_dt;
        step(_dt,S,states);
        //S->relax();

        fwrite(&t,sizeof(double),1,f);
        S->write(f);  //write particles info after interaction in this time step

        if(diagnose){
            S->diagnose();
            fwrite(&t,sizeof(double),1,fdiag);
            S->writediagnosis(fdiag);
        }
    }
    fclose(f);
    double tEnd = TIME();
    OUT("Total Time (s)",tEnd - tStart);
    gsl_vector_free(states);
}

void pawan::__integration::integrate(__system *S,
                                     __io *IO,
                                     NetworkInterfaceTCP<OPawanRecvData,OPawanSendData> *networkCommunicatorTest,
                                     bool diagnose){
    double tStart = TIME();
    gsl_vector *states = gsl_vector_calloc(S->_totalmaxsize); //ip: have the *rates pointer here outside too for speed up
    FILE *fdiag = IO->create_binary_file(".diagnosis");
    OPawanRecvData opawanrecvdata;
    networkCommunicatorTest->getrecieveBuffer(opawanrecvdata);
    _t = opawanrecvdata.t;
    FILE *f = IO->create_binary_file(".wake");
    if(diagnose) {
        S->writenu(fdiag);
        S->diagnose();
        fwrite(&_t,sizeof(double),1,fdiag);
        S->writediagnosis(fdiag);
    }

    size_t stepnum = 0;
    while(_t <= opawanrecvdata.tfinal) {
        OUT("\tTime", _t);
        OUT("\tStepNum", stepnum);
        gsl_vector_set_zero(states);
        S->getStates(states);
        step(opawanrecvdata.deltat, S, states);
        S->split(stepnum);
        S->merge(stepnum);
        if (diagnose) {
            S->diagnose();
            fwrite(&_t, sizeof(double), 1, fdiag);
            S->writediagnosis(fdiag);
        }
        int transient_steps = opawanrecvdata.transientsteps;
        if (stepnum < transient_steps) {
            printf("Vinf = %3.2e, %3.2e, %3.2e \n", opawanrecvdata.Vinf[0], opawanrecvdata.Vinf[1], opawanrecvdata.Vinf[2]);
            opawanrecvdata.Vinf[2] = opawanrecvdata.Vinf[2] + 15 * (transient_steps - stepnum) / transient_steps;
            printf("Vinf + suppress = %3.2e, %3.2e, %3.2e", opawanrecvdata.Vinf[0], opawanrecvdata.Vinf[1],
                   opawanrecvdata.Vinf[2]);
        }
        S->updateVinfEffect(opawanrecvdata.Vinf,opawanrecvdata.deltat);
        S->updateBoundVorEffect(&opawanrecvdata,opawanrecvdata.deltat);
        fwrite(&_t,sizeof(double),1,f);
        S->write(f);  //write particles info after interaction in this time step

        OPawanSendData opawansenddata;//create it once outside the loop and should be good
        S->getInflow(&opawanrecvdata,&opawansenddata);
        networkCommunicatorTest->send_data(opawansenddata);

        //S->diagnose();
        stepnum = stepnum+1;
        if(_t <= (opawanrecvdata.tfinal - 1*opawanrecvdata.deltat)){ //run till end of dymore sim
            networkCommunicatorTest->recieve_data(opawanrecvdata);
            S->addParticles(&opawanrecvdata, stepnum);
            _t = opawanrecvdata.t;
        }
        else
            break;

    }
    fclose(f);
    double tEnd = TIME();
    OUT("Total Time (s)",tEnd - tStart);
    gsl_vector_free(states);
}

void pawan::__integration::step(const double &dt, __system *S, gsl_vector* states){
    gsl_vector *rates = gsl_vector_calloc(S->_totalmaxsize);  //ip: large memory allocations would make code slow here
    S->solve();
    S->getRates(rates);
    gsl_vector_scale(rates,dt);
    printf("Position change of 1st particle     __integration::step(): %+8.3e, %+8.3e, %+8.3e\n",
           gsl_vector_get(rates, 0),gsl_vector_get(rates, 1),gsl_vector_get(rates, 2));
    gsl_vector_add(states,rates);
    S->setStates(states);
    gsl_vector_free(rates);
}

