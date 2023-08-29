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
#include "src/networkinterface/networkdatastructures.h"
#include "src/networkinterface/networkinterface.h"
#include "src/networkinterface/networkinterface.cpp" //templates included this way
#include "interaction/gpu.cuh"
#include "integration/gpu_euler.cuh"

#define OUTPUTIP "127.0.0.1"
#define NETWORKBUFFERSIZE 50
#define PORT 8899

void measure(pawan::__interaction *pInteraction) {

    int iterations = 2;

    auto t1 = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < iterations; i++){
        pInteraction->solve();
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "average time: "
              << std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count() / iterations
              << " microseconds\n";
}

int main(int argc, char* argv[]){

    std::cout << std::setprecision(16) << std::scientific;
    PAWAN();

    //test();

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
    //pawan::__wake *W1 = new pawan::__vring(1.0,0.1,4,80,0.1);
    //pawan::__io *IOvring1 = new pawan::__io("vring4by80_1");
    //pawan::__wake *W2 = new pawan::__vring(1.0,0.1,4,80,0.1);
    //pawan::__io *IOvring2 = new pawan::__io("vring4by80_2");
    //pawan::__io *IOvrings = new pawan::__io("vring4by80_vring4by80_eulerfusion");
    //pawan::__wake *W1 = new pawan::__vring(1.0,0.1,5,100,0.0840);
    //pawan::__io *IOvring1 = new pawan::__io("vring5by100_1");
    //pawan::__wake *W2 = new pawan::__vring(1.0,0.1,5,100,0.0840);
    //pawan::__io *IOvring2 = new pawan::__io("vring5by100_2");
    //pawan::__io *IOvrings = new pawan::__io("vring5by100_vring5by100_eulerfusion");
    pawan::__wake *W1 = new pawan::__vring(1.0,0.1,3,49,0.1924);
    pawan::__io *IOvring1 = new pawan::__io("vring3by49_1");
    pawan::__wake *W2 = new pawan::__vring(1.0,0.1,3,49,0.1924);
    pawan::__io *IOvring2 = new pawan::__io("vring3by49_2");
    pawan::__io *IOvrings = new pawan::__io("vring3by49_vring3by49_eulerfusion");

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
    Wvring1->rotate(1,M_1_PI/6);  //rotate about y-axis by 15 deg
    Wvring2->rotate(1,-M_1_PI/6); //rotate about y-axis by -15 deg
    double translate_vec[3]={2.7,0.,0.};
    Wvring2->translate(translate_vec);

    //pawan::__interaction *Svring = new pawan::__interaction(Wvring);
    pawan::__interaction *Svring = new pawan::gpu(Wvring1,Wvring2);

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
    pawan::__wake *W1 = new pawan::__vring(1.0,0.1,4,80,0.1);
    pawan::__io *IOvring1 = new pawan::__io("vring4by80_1");
    pawan::__wake *W2 = new pawan::__vring(1.0,0.1,4,80,0.1);
    pawan::__io *IOvring2 = new pawan::__io("vring4by80_2");
    pawan::__interaction *S1 = new pawan::__parallel(W1);
    pawan::__interaction *S2 = new pawan::__parallel(W2);
    pawan::__resolve *R = new pawan::__resolve();
    R->rebuild(S1,IOvring1);printf("resolved ring 1 \n");
    R->rebuild(S2,IOvring2);printf("resolved ring 1 \n");
    pawan::__wake *Wvring1 = new pawan::__wake(W1);
    pawan::__wake *Wvring2 = new pawan::__wake(W2);
    Wvring1->rotate(1,M_1_PI/4); Wvring2->rotate(1,-M_1_PI/4);
    double translate_vec[3]={2.7,0.,0.};Wvring2->translate(translate_vec);
    pawan::__interaction *Svring = new pawan::__parallel(Wvring1,Wvring2);
    pawan::__integration *INvring = new pawan::gpu_euler<>(1,200);
    pawan::__io *IOvrings = new pawan::__io("vring4by80_1and2_fissionfusion");
    INvring->integrate(Svring,IOvrings,false);
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

/*
    //%%%%%%%%%%%%%%      isolated ring     %%%%%%%%%%%%%%%%
   // pawan::__wake *W = new pawan::__vring(1.0,0.1,1,10,0.1);
  //  pawan::__io *IOvring = new pawan::__io("vring4by80_euler_gpu");
    //pawan::__wake *W = new pawan::__vring(1.0,0.1,5,100,0.0840);
    //pawan::__io *IOvring = new pawan::__io("vring_5by100");
    pawan::__wake *W = new pawan::__vring(1.0,0.1,6,117,0.0735);
    pawan::__io *IOvring = new pawan::__io("vring_6by117_gpu_euler");

    //pawan::__interaction *S = new pawan::__interaction(W);
    pawan::__interaction *S = new pawan::__parallel(W);

    pawan::__resolve *R = new pawan::__resolve();
    S->diagnose();//simply calculate diagnostics
    R->rebuild(S,IOvring);
    W->print();
    S->diagnose();
    S->solve();
    W->print();


    //pawan::__io *IOvring = new pawan::__io("ring");


    //pawan::__wake *W = new pawan::__ring(1.0,5.0,0.1,5);
    pawan::__wake *Wvring = new pawan::__wake(W);
    //pawan::__interaction *Svring = new pawan::__interaction(Wvring);
    pawan::__interaction *Svring = new pawan::__parallel(Wvring);
    pawan::__integration *INvring = new pawan::gpu_euler(5,100);
    //pawan::__integration *INvring = new pawan::__rk4(5,100);

    INvring->integrate(Svring,IOvring,false);


    //delete R;
    //delete S;
    delete W;
    delete Wvring;
    delete Svring;
    delete INvring;
    delete IOvring;
*/

    std::cout << "Setup\n";
    unsigned long int seed1 = 17478738;

    int size1 = 20000;

    gsl_rng * r;
    const gsl_rng_type * T;

    gsl_rng_env_setup();
    T = gsl_rng_default;
    r = gsl_rng_alloc (T);

    {
        //warmup so that GPU can initialise
        {
            gsl_rng_set(r, seed1);
            pawan::test_wake wakeGPU = pawan::test_wake(size1, r);
            pawan::__interaction *interactionGPU = new pawan::gpu<>(&wakeGPU);
            interactionGPU->solve();
            delete interactionGPU;
        }



        {
            constexpr int threadblocks = 128, unrollFactor = 1;
            std::cout << "Testing with threadsblocks: " << threadblocks << " and unrollFactor: " << unrollFactor
                      << "\n";
            gsl_rng_set(r, seed1);
            pawan::test_wake wakeGPU = pawan::test_wake(size1, r);
            pawan::__interaction *interactionGPU = new pawan::gpu<threadblocks, unrollFactor>(&wakeGPU);
            measure(interactionGPU);
            delete interactionGPU;
        }

        {
            constexpr int threadblocks = 64, unrollFactor = 1;
            std::cout << "Testing with threadsblocks: " << threadblocks << " and unrollFactor: " << unrollFactor
                      << "\n";
            gsl_rng_set(r, seed1);
            pawan::test_wake wakeGPU = pawan::test_wake(size1, r);
            pawan::__interaction *interactionGPU = new pawan::gpu<threadblocks, unrollFactor>(&wakeGPU);
            measure(interactionGPU);
            delete interactionGPU;
        }

        {
            constexpr int threadblocks = 64, unrollFactor = 2;
            std::cout << "Testing with threadsblocks: " << threadblocks << " and unrollFactor: " << unrollFactor
                      << "\n";
            gsl_rng_set(r, seed1);
            pawan::test_wake wakeGPU = pawan::test_wake(size1, r);
            pawan::__interaction *interactionGPU = new pawan::gpu<threadblocks, unrollFactor>(&wakeGPU);
            measure(interactionGPU);
            delete interactionGPU;
        }
    }

    return EXIT_SUCCESS;

}


