#ifndef PAWAN_NETWORKDATASTRUCTURE_H
#define PAWAN_NETWORKDATASTRUCTURE_H

#include <iostream>
#define PAWAN_MAXLFNLINES  4    //assuming only one rotor or wing entity
                                //(formulation not guaranteed to work with more than 1 rotors or rotor+wing)
#define PAWAN_MAXAST       200  //per lfnline

typedef enum pawancinttype { //circ interpolation type
    CINT_ANY, CINT_CONSTANT, CINT_LINEAR
} Pawancinttype;

typedef enum pawanregfunctype {//regularisation function for the kernel
    REGF_ANY, REGF_GAUSSIAN
} Pawanregfunctype;

typedef struct pawanrecvdata {    //better implementation possible perhaps
    double Vinf[3]; //far field flow velocity vector
    int NbOfLfnLines; //number of lifting lines (flap lfnlines not counted)
    int NbOfAst[PAWAN_MAXLFNLINES];
    Pawancinttype pawancinttype;
    Pawanregfunctype pawanregfunctype;
    double hres;
    double acrossa;
    double TEpos_prev[PAWAN_MAXLFNLINES*PAWAN_MAXAST*3]; //max 4 blades, 200 airstations per blade, 3 coordinates
    double circ_prev[PAWAN_MAXLFNLINES*PAWAN_MAXAST];
    double TEpos[PAWAN_MAXLFNLINES*PAWAN_MAXAST*3];
    double circ[PAWAN_MAXLFNLINES*PAWAN_MAXAST];
    double astpos[PAWAN_MAXLFNLINES*PAWAN_MAXAST*3];

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
