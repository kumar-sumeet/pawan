#ifndef PAWAN_NETWORKDATASTRUCTURE_H
#define PAWAN_NETWORKDATASTRUCTURE_H

#include <iostream>
#define PAWAN_MAXLFNLINES  4    //assuming only one rotor or wing entity
                                //(formulation not guaranteed to work with more than 1 rotors or rotor+wing)
#define PAWAN_MAXAST       200  //per lfnline
#define W_NAME 128

typedef enum pawancinttype { //circ interpolation type
    CINT_ANY, CINT_CONSTANT, CINT_LINEAR
} Pawancinttype;

typedef enum pawanregfunctype {//regularisation function for the kernel
    REGF_ANY, REGF_GAUSSIAN
} Pawanregfunctype;

typedef struct pawanrecvdata {    //better implementation possible perhaps
    char Dymfilename[2*W_NAME];
    double Vinf[3]; //far field flow velocity vector
    int NbOfLfnLines; //number of lifting lines (flap lfnlines not counted)
    int NbOfAst[PAWAN_MAXLFNLINES];
    double lfnlen[PAWAN_MAXLFNLINES]; //span of lfnline
    Pawancinttype pawancinttype;
    Pawanregfunctype pawanregfunctype;
    int trailvor;
    int shedvor;
    int suprootvor; // 0 -> do not suppress root vortices, 1-> suppress root vortices
    int transientsteps;
    int infl2D;  //0-> full 3D wake used to calculate inflow; 1-> only panel shed wake used to calculate inflow
    int spanres;
    double acrossa;
    double deltat;
    double t; //time
    double tfinal; //final dymore simulation time
    double span_disc[PAWAN_MAXLFNLINES*PAWAN_MAXAST]; //non-dimensional location of blade spanwise discretization
    double TEpos_prev[PAWAN_MAXLFNLINES*PAWAN_MAXAST*3]; //max 1 rotor, 4 blades, 200 airstations per blade, 3 coordinates
    double circ_prev[PAWAN_MAXLFNLINES*PAWAN_MAXAST];
    double TEpos[PAWAN_MAXLFNLINES*PAWAN_MAXAST*3];
    double circ[PAWAN_MAXLFNLINES*PAWAN_MAXAST];
    double astpos[PAWAN_MAXLFNLINES*PAWAN_MAXAST*3];
    double astvel[PAWAN_MAXLFNLINES*PAWAN_MAXAST*3];  //ip: this should actually be TEvel
} *PawanRecvData, OPawanRecvData;

typedef struct pawansenddata{
    double lambda[PAWAN_MAXLFNLINES*PAWAN_MAXAST*3];
} *PawanSendData, OPawanSendData;


#define PawanRecvGetTEPos(pawanrecvdata,iii)\
        pawanrecvdata->TEpos[iii]
#define PawanRecvGetTEPosprev(pawanrecvdata,iii)\
        pawanrecvdata->TEpos_prev[iii]
#define PawanRecvGetCirc(pawanrecvdata,iii)\
        pawanrecvdata->circ[iii]
#define PawanRecvGetAstPos(pawanrecvdata,iii)\
        pawanrecvdata->astpos[iii]

#endif //PAWAN_NETWORKDATASTRUCTURE_H
