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
#include "integration/gpu_int.cuh"
#include "test.cuh"

#define OUTPUTIP "127.0.0.1"
#define NETWORKBUFFERSIZE 50

int main(int argc, char* argv[]){

    std::cout << std::setprecision(16) << std::scientific;
    PAWAN();
    //%%%%%%%%%%%%     Dymore coupling    %%%%%%%%%%%%%%%%%%
    int port = 8899;
    if (argc==2) {
        char *cli_port = argv[1];
        port = atoi(cli_port);
    }
    NetworkInterfaceTCP<OPawanRecvData, OPawanSendData>
            networkCommunicatorTest(port, OUTPUTIP, port, NETWORKBUFFERSIZE, true);
    networkCommunicatorTest.socket_init();
    OPawanRecvData opawanrecvdata;
    networkCommunicatorTest.recieve_data(opawanrecvdata);
    PawanRecvData pawanrecvdata = &opawanrecvdata;
    std::string dymfilename = pawanrecvdata->Dymfilename;
    pawan::__io *IOdym = new pawan::__io(dymfilename);
    pawan::__wake *W = new pawan::__wake(pawanrecvdata);
    //pawan::__interaction *S = new pawan::__interaction(W);
    pawan::__interaction *S = new pawan::__parallel(W);
    //pawan::__integration *IN = new pawan::__integration();
    pawan::__integration *IN = new pawan::gpu_int<>();
    IN->integrate(S,IOdym,&networkCommunicatorTest,false);
    delete IN;
    delete S;
    delete IOdym;

/*
    //%%%%%%%%%%%%     Fusion rings    %%%%%%%%%%%%%%%%%%
    pawan::__io *IOvring = new pawan::__io("vring3by64");
    FILE *fvringtemp = IOvring->open_binary_file(".vringwake");
    int numparticles;fread(&numparticles,sizeof(size_t),1,fvringtemp);fclose(fvringtemp);
    FILE *fvring = IOvring->open_binary_file(".vringwake");
    pawan::__wake *Wvringtemp = new pawan::__wake(numparticles, fvring);//temp needed in order to use gsl_*_fread()
    pawan::__wake *Wvring1 = new pawan::__wake(Wvringtemp);
    pawan::__wake *Wvring2 = new pawan::__wake(Wvringtemp);

    pawan::__io *IOvrings = new pawan::__io("vring3by64vring3by64fusion_rk4");

    Wvring1->rotate(1,-M_PI/12);  //rotate about y-axis by -15 deg
    Wvring2->rotate(1,M_PI/12); //rotate about y-axis by 15 deg
    double translate_vec1[3]={1.35,0.,0.}, translate_vec2[3]={-1.35,0.,0.};
    Wvring1->translate(translate_vec1);
    Wvring2->translate(translate_vec2);

    pawan::__interaction *Svring = new pawan::__interaction(Wvring1,Wvring2);

    pawan::__integration *INvring = new pawan::gpu_int<>(8,160);
    INvring->integrate(Svring,IOvrings,true);

    delete Svring;
    delete INvring;
    delete Wvring1;
    delete Wvring2;
    delete IOvrings;
*/
/*
    //%%%%%%%%%%%%     Fission-Fusion rings    %%%%%%%%%%%%%%%%%%
    pawan::__io *IOvring = new pawan::__io("vring2by52");
    FILE *fvringtemp = IOvring->open_binary_file(".vringwake");
    int numparticles;fread(&numparticles,sizeof(size_t),1,fvringtemp);fclose(fvringtemp);
    FILE *fvring = IOvring->open_binary_file(".vringwake");
    pawan::__wake *Wvringtemp = new pawan::__wake(numparticles, fvring);//temp needed in order to use gsl_*_fread()
    pawan::__wake *Wvring1 = new pawan::__wake(Wvringtemp);
    pawan::__wake *Wvring2 = new pawan::__wake(Wvringtemp);
    pawan::__io *IOvrings = new pawan::__io("vring2by52vring2by52fissionfusion_rk4");

    Wvring1->rotate(1,-M_PI/6); Wvring2->rotate(1,M_PI/6);
    double translate_vec1[3]={1.5,0.,0.},translate_vec2[3]={-1.5,0.,0.};
    Wvring1->translate(translate_vec1);Wvring2->translate(translate_vec2);
    pawan::__interaction *Svring = new pawan::__interaction(Wvring1,Wvring2);
    pawan::__integration *INvring = new pawan::gpu_int<>(30,400);
    INvring->integrate(Svring,IOvrings,true);
    delete Svring;delete INvring;
    delete Wvring1;delete Wvring2;delete IOvrings;
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
/*        //%%%%%%%%%%%%%%      knot ring     %%%%%%%%%%%%%%%%
    pawan::__io *IOvring = new pawan::__io("vring0by175");
    FILE *fvringtemp = IOvring->open_binary_file(".vringwake");
    int numparticles;fread(&numparticles,sizeof(size_t),1,fvringtemp);fclose(fvringtemp);
    FILE *fvring = IOvring->open_binary_file(".vringwake");
    pawan::__wake *Wvringtemp = new pawan::__wake(numparticles, fvring);//temp needed in order to use gsl_*_fread()
    pawan::__wake *Wvring1 = new pawan::__wake(Wvringtemp);
    pawan::__wake *Wvring2 = new pawan::__wake(Wvringtemp);
    pawan::__io *IOvrings = new pawan::__io("vring0by175vring0by175knot_rk4");

    Wvring1->rotate(1,M_PI/2);
    double translate_vec1[3]={0.0,1.0,0.};
    Wvring1->translate(translate_vec1);
    pawan::__interaction *Svring = new pawan::__interaction(Wvring1,Wvring2);
    //pawan::__interaction *Svring = new pawan::__interaction(Wvring1);
    pawan::__integration *INvring = new pawan::gpu_int<>(30,1200);
    INvring->integrate(Svring,IOvrings,true);
    delete Svring;delete INvring;
    delete Wvring1;delete Wvring2;delete IOvrings;
*/

    //%%%%%%%%%%%%%%      isolated ring     %%%%%%%%%%%%%%%%
    //pawan::__io *IOvring = new pawan::__io("vring3by64");
    //pawan::__io *IOvring = new pawan::__io("vring2by52");
    //pawan::__wake *W = new pawan::__vring(1.0,0.1,4,80,0.1);
    //pawan::__io *IOvring = new pawan::__io("vring4by80");
    //pawan::__wake *W = new pawan::__vring(1.0,0.1,5,100,0.0840);
    //pawan::__io *IOvring = new pawan::__io("vring5by100");
    //pawan::__wake *W = new pawan::__vring(1.0,0.1,6,117,0.0735);
    pawan::__io *IOvring = new pawan::__io("vring6by117");
    //pawan::__io *IOvring = new pawan::__io("vring0by175");

/*
    FILE *fvringtemp = IOvring->open_binary_file(".vringwake");
    int numparticles;fread(&numparticles,sizeof(size_t),1,fvringtemp);fclose(fvringtemp);
    FILE *fvring = IOvring->open_binary_file(".vringwake");
    pawan::__wake *Wvringtemp = new pawan::__wake(numparticles, fvring);//temp needed in order to use gsl_*_fread()
    pawan::__wake *Wvring = new pawan::__wake(Wvringtemp);


    pawan::__interaction *Svring = new pawan::__interaction(Wvring);
    //pawan::__interaction *Svring = new pawan::__parallel(Wvring);
    pawan::__integration *INvring = new pawan::gpu_int<>(5, 100);
    //pawan::__integration *INvring = new pawan::gpu_int<>(0.025, 1);
    //pawan::__integration *INvring = new pawan::__integration(0.025,1);
    //pawan::__integration *INvring = new pawan::__rk4(5,100);

    INvring->integrate(Svring,IOvring,true);

    delete Wvring;
    delete Svring;
    delete INvring;
    delete IOvring;
*/
    return EXIT_SUCCESS;

}