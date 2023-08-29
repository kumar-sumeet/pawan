/*! Particle Wake Analysis
 * \file pawan.cpp
 * \brief Main executable code
 * @author Puneet Singh
 * @date 03/28/2021
 */

#include <iostream>
#include <iomanip> // Required for set precision
#include <gsl/gsl_rng.h>


#include "utils/print_utils.h"
#include "io/io.h"
#include "wake/wake.h"
#include "wake/ring.h"
#include "wake/square.h"
#include "wake/vring.h"
#include "wake/test_wake.h"
#include "src/interaction/interaction.h"
#include "src/interaction/parallel.h"
#include "src/integration/integration.h"
#include "src/resolve/resolve.h"
#include "src/integration/rk4.h"
#include "src/integration/rk3.h"
#include "src/networkinterface/networkdatastructures.h"
#include "src/networkinterface/networkinterface.h"
#include "src/networkinterface/networkinterface.cpp" //templates included this way
#include "interaction/gpu.cuh"
#include "integration/gpu_euler.cuh"
#include "test.cuh"

#define OUTPUTIP "127.0.0.1"
#define NETWORKBUFFERSIZE 50
#define PORT 8899

int main(int argc, char* argv[]){

    std::cout << std::setprecision(16) << std::scientific;
    PAWAN();
/*
    //%%%%%%%%%%%%     Dymore coupling    %%%%%%%%%%%%%%%%%%
    NetworkInterfaceTCP<OPawanRecvData,OPawanSendData>
            networkCommunicatorTest(PORT, OUTPUTIP, PORT, NETWORKBUFFERSIZE, true);
    networkCommunicatorTest.socket_init();
    OPawanRecvData opawanrecvdata;
    networkCommunicatorTest.recieve_data(opawanrecvdata);
    PawanRecvData pawanrecvdata = &opawanrecvdata;
    std::string dymfilename = pawanrecvdata->Dymfilename;
    pawan::__io *IOdym = new pawan::__io(dymfilename);
    pawan::__wake *W = new pawan::__wake(pawanrecvdata);
    //pawan::__interaction *S = new pawan::__interaction(W);
    pawan::__interaction *S = new pawan::__parallel(W);
    pawan::__integration *IN = new pawan::__integration();
    IN->integrate(S,IOdym,&networkCommunicatorTest,false);
    delete IN;
    delete S;
    delete IOdym;
*/
/*
    //%%%%%%%%%%%%     Fusion rings    %%%%%%%%%%%%%%%%%%
    pawan::__wake *W1 = new pawan::__vring(1.0,0.1,3,49,0.1924);
    pawan::__io *IOvring1 = new pawan::__io("vring3by49_1");
    pawan::__wake *W2 = new pawan::__vring(1.0,0.1,3,49,0.1924);
    pawan::__io *IOvring2 = new pawan::__io("vring3by49_2");
    pawan::__io *IOvrings = new pawan::__io("vring3by49vring3by49fusion_rk4");

    //pawan::__interaction *S = new pawan::__interaction(W1);
    pawan::__interaction *S1 = new pawan::gpu(W1);
    pawan::__interaction *S2 = new pawan::gpu(W2);

    pawan::__resolve *R = new pawan::__resolve();
    R->rebuild(S1,IOvring1);
    printf("resolved ring 1 \n");
    R->rebuild(S2,IOvring2);//ip: *.wakeinfluence from above gets overwritten here
    printf("resolved ring 2 \n");

    pawan::__wake *Wvring1 = new pawan::__wake(W1);
    pawan::__wake *Wvring2 = new pawan::__wake(W2);
    Wvring1->rotate(1,M_1_PI/12);  //rotate about y-axis by 15 deg
    Wvring2->rotate(1,-M_1_PI/12); //rotate about y-axis by -15 deg
    double translate_vec[3]={2.7,0.,0.};
    Wvring2->translate(translate_vec);

    //pawan::__interaction *Svring = new pawan::__interaction(Wvring);
    pawan::__interaction *Svring = new pawan::gpu<>(Wvring1,Wvring2);
    //pawan::__interaction *Svring = new pawan::__parallel(Wvring1,Wvring2);

    //relaxed -diverges at 196 steps, normal - diverges at 300
    pawan::__integration *INvring = new pawan::__integration(15,300);
    //pawan::__integration *INvring = new pawan::__integration(9.75,195);
    //pawan::__integration *INvring = new pawan::__rk4(0.01,1);
    //pawan::__integration *INvring = new pawan::__rk4(25,500);

    INvring->integrate(Svring,IOvrings,true);

    delete Svring;
    delete INvring;

    delete R;
    delete S1;
    delete S2;
    delete W1;
    delete W2;
    delete Wvring1;
    delete Wvring2;
    delete IOvring1;
    delete IOvring2;
    delete IOvrings;
*/
/*
    //%%%%%%%%%%%%     Fission-Fusion rings    %%%%%%%%%%%%%%%%%%
    pawan::__wake *W1 = new pawan::__vring(1.0,0.125,2,52,0.1562);
    pawan::__io *IOvring1 = new pawan::__io("vring2by52_1");
    pawan::__wake *W2 = new pawan::__vring(1.0,0.125,2,52,0.1562);
    pawan::__io *IOvring2 = new pawan::__io("vring2by52_2");
    pawan::__io *IOvrings = new pawan::__io("vring2by52vring2by52fissionfusion_rk4");

    pawan::__interaction *S1 = new pawan::__parallel(W1);
    pawan::__interaction *S2 = new pawan::__parallel(W2);
    pawan::__resolve *R = new pawan::__resolve();
    R->rebuild(S1,IOvring1);printf("resolved ring 1 \n");
    R->rebuild(S2,IOvring2);printf("resolved ring 1 \n");
    pawan::__wake *Wvring1 = new pawan::__wake(W1);
    pawan::__wake *Wvring2 = new pawan::__wake(W2);
    Wvring1->rotate(1,M_1_PI/6); Wvring2->rotate(1,-M_1_PI/6);
    double translate_vec[3]={3.0,0.,0.};Wvring2->translate(translate_vec);
    pawan::__interaction *Svring = new pawan::__parallel(Wvring1,Wvring2);
    pawan::__integration *INvring = new pawan::__rk4(30,600);
    INvring->integrate(Svring,IOvrings,true);
    delete Svring;delete INvring;delete R;delete S1;delete S2;delete W1;delete W2;
    delete Wvring1;delete Wvring2;delete IOvring1;delete IOvring2;delete IOvrings;
*/
/*
    pawan::__interaction *S = new pawan::__interaction(W1,W2);
    pawan::__integration *IN = new pawan::__rk4(30,600);
    IN->integrate(S,IO,&networkCommunicatorTest);

    //Leap-frogging rings
    pawan::__wake *W1 = new pawan::__ring(8.0,10.0,0.1,100);
    pawan::__wake *W2 = new pawan::__ring(8.0,10.0,0.1,100);
    double translate_vec[3]={0.,0.,-3.};
    W2->translate(translate_vec);
    pawan::__interaction *S = new pawan::__interaction(W1,W2);
    pawan::__integration *IN = new pawan::__rk4(30,600);
    IN->integrate(S,IO,&networkCommunicatorTest);
*/


    //%%%%%%%%%%%%%%      isolated ring     %%%%%%%%%%%%%%%%
    //pawan::__wake *W = new pawan::__vring(1.0,0.1,4,80,0.1);
    //pawan::__io *IOvring = new pawan::__io("vring4by80_euler_gpu");
    //pawan::__wake *W = new pawan::__vring(1.0,0.1,5,100,0.0840);
    //pawan::__io *IOvring = new pawan::__io("vring_5by100");
    pawan::__wake *W = new pawan::__vring(1.0,0.1,6,117,0.0735);
    pawan::__io *IOvring = new pawan::__io("vring_6by117_gpu");

    //pawan::__interaction *S = new pawan::__interaction(W);
    pawan::__interaction *S = new pawan::__parallel(W);

    pawan::__resolve *R = new pawan::__resolve();
    S->diagnose();//simply calculate diagnostics
    R->rebuild(S,IOvring);
    W->print();
    S->diagnose();
    S->solve();
    W->print();

    pawan::__wake *Wvring = new pawan::__wake(W);
    //pawan::__interaction *Svring = new pawan::__interaction(Wvring);
    pawan::__interaction *Svring = new pawan::__parallel(Wvring);
    pawan::__integration *INvring = new pawan::gpu_euler<>(5,100);
    //pawan::__integration *INvring = new pawan::__rk4(5,100);

    INvring->integrate(Svring,IOvring,true);

    delete R;
    delete S;
    delete W;
    delete Wvring;
    delete Svring;
    delete INvring;
    delete IOvring;

    return EXIT_SUCCESS;

}