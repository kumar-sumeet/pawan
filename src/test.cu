
#include "test.cuh"
#include <iostream>

#include "wake/wake.h"
#include "wake/test_wake.h"
#include <gsl/gsl_vector_double.h>
#include "interaction/interaction_utils.h"
#include "interaction/interaction_utils_gpu.cuh"
#include "interaction/interaction.h"
#include "interaction/gpu.cuh"
#include <gsl/gsl_rng.h>
#include "io/io.h"
#include "integration/gpu_euler.cuh"
#include "wake/ring.h"
#include "wake/vring.h"
#include "resolve/resolve.h"

using namespace pawan;

//constexpr double epsilon = 1e-13;
constexpr double epsilonsToTest[]{
        1e-10,
        1e-11,
        1e-12,
        1e-13,
        1e-14
};

//output errors with location
bool closeToZero(double gpuV);

__global__ void testKernel(double nu,
                           const double4 *data,
                           double3 *returnVals){
    INTERACT_GPU(nu, data[0], data[2], data[1], data[3], returnVals[0], returnVals[1]);
}

void measure(pawan::__interaction *pInteraction) {

    int iterations = 20;

    auto t1 = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < iterations; i++){
        pInteraction->solve();
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "average time: "
              << ((double) std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count()) / iterations
              << " milliseconds\n";
}


bool testSingleInteract(double nu, double s_src, double s_trg, gsl_vector *r_src, gsl_vector *r_trg, gsl_vector *a_src,
                        gsl_vector *a_trg, double v_src, double v_trg) {

    //Outputs of CPU version
    double vx_s = 0.0, vy_s = 0.0, vz_s = 0.0;
    double qx_s = 0.0, qy_s = 0.0, qz_s = 0.0;
    gsl_vector *dr_trg = gsl_vector_alloc(3);
    gsl_vector *da_trg = gsl_vector_alloc(3);    //these are the exact negative -> ignore


    INTERACT(nu, s_src, s_trg, r_src, r_trg, a_src, a_trg, v_src, v_trg, dr_trg, da_trg, vx_s, vy_s, vz_s, qx_s, qy_s, qz_s);

    //std::cout << "CPU velocity " << vx_s << ", " << vy_s << ", " << vz_s << "\n";
    // std::cout << "CPU velocity' " << gsl_vector_get(dr_trg,0) << ", " << gsl_vector_get(dr_trg,1)<< ", " << gsl_vector_get(dr_trg,2) << "\n";
    //std::cout << "CPU retvorticity " << qx_s << ", " << qy_s << ", " << qz_s << "\n";

    //copy values to the format used by the gpu function
    double4 *data;
    double3 *retVals;

    checkGPUError(cudaMallocManaged(&data,4* sizeof(double4)));
    checkGPUError(cudaMallocManaged(&retVals,2* sizeof(double3)));

    data[0].x = gsl_vector_get(r_src, 0);
    data[0].y = gsl_vector_get(r_src, 1);
    data[0].z = gsl_vector_get(r_src, 2);
    data[0].w = s_src;

    data[1].x = gsl_vector_get(a_src, 0);
    data[1].y = gsl_vector_get(a_src, 1);
    data[1].z = gsl_vector_get(a_src, 2);
    data[1].w = v_src;

    data[2].x = gsl_vector_get(r_trg, 0);
    data[2].y = gsl_vector_get(r_trg, 1);
    data[2].z = gsl_vector_get(r_trg, 2);
    data[2].w = s_trg;

    data[3].x = gsl_vector_get(a_trg, 0);
    data[3].y = gsl_vector_get(a_trg, 1);
    data[3].z = gsl_vector_get(a_trg, 2);
    data[3].w = v_trg;

    testKernel<<<1,1>>>(nu,data,retVals);
    checkGPUError(cudaDeviceSynchronize());

    //std::cout << "GPU velocity " << retVals[0].x << ", " << retVals[0].y << ", " << retVals[0].z << "\n";
    //std::cout << "GPU retvorticity " << retVals[1].x << ", " << retVals[1].y << ", " << retVals[1].z << "\n";


    bool equal = gsl_fcmp(retVals[0].x, vx_s, epsilonsToTest[0]) == 0
                 && gsl_fcmp(retVals[0].y, vy_s, epsilonsToTest[0]) == 0
                 && gsl_fcmp(retVals[0].z, vz_s, epsilonsToTest[0]) == 0
                 && gsl_fcmp(retVals[1].x, qx_s, epsilonsToTest[0]) == 0
                 && gsl_fcmp(retVals[1].y, qy_s, epsilonsToTest[0]) == 0
                 && gsl_fcmp(retVals[1].z, qz_s, epsilonsToTest[0]) == 0;

    checkGPUError(cudaFree(data));
    checkGPUError(cudaFree(retVals));
    gsl_vector_free(dr_trg);
    gsl_vector_free(da_trg);

    return equal;
}

//e-13: 0.18 %
//e-15: 2.12 %
//e-17: 9.76 %
void testInteractWithRandomValues() {
    gsl_vector *r_src = gsl_vector_alloc(3); //Position
    gsl_vector *a_src = gsl_vector_alloc(3); //vorticity
    gsl_vector *r_trg = gsl_vector_alloc(3);
    gsl_vector *a_trg = gsl_vector_alloc(3);

    gsl_rng * r;
    const gsl_rng_type * T;

    gsl_rng_env_setup();
    T = gsl_rng_default;
    r = gsl_rng_alloc (T);

    int wrong = 0;
    int iterations = 1000000;

    for(int i = 0; i < iterations; i++) {

        double nu = 2.5e-3;

        gsl_vector_set(r_trg, 0, gsl_rng_uniform(r) * 6 -3);
        gsl_vector_set(r_trg, 1, gsl_rng_uniform(r) * 6 -3);
        gsl_vector_set(r_trg, 2, gsl_rng_uniform(r) * 6 -3);

        gsl_vector_set(a_trg, 0, gsl_rng_uniform(r) * 6 -3);
        gsl_vector_set(a_trg, 1, gsl_rng_uniform(r) * 6 -3);
        gsl_vector_set(a_trg, 2, gsl_rng_uniform(r) * 6 -3);

        gsl_vector_set(r_src, 0, gsl_rng_uniform(r) * 6 -3);
        gsl_vector_set(r_src, 1, gsl_rng_uniform(r) * 6 -3);
        gsl_vector_set(r_src, 2, gsl_rng_uniform(r) * 6 -3);

        gsl_vector_set(a_src, 0, gsl_rng_uniform(r) * 6 -3);
        gsl_vector_set(a_src, 1, gsl_rng_uniform(r) * 6 -3);
        gsl_vector_set(a_src, 2, gsl_rng_uniform(r) * 6 -3);

        //smoothing radius and volume
        double s_trg = gsl_rng_uniform(r);
        double v_trg = gsl_rng_uniform(r);
        double s_src = gsl_rng_uniform(r);
        double v_src = gsl_rng_uniform(r);

        if(!testSingleInteract(nu, s_src, s_trg, r_src, r_trg, a_src, a_trg, v_src, v_trg)){
            /*std::cout << "Number "<< i << " is wrong! nu:" << nu << " s: " << s_src << ", " << s_trg << " v: " <<v_src << ", " << v_trg << "\n";
            OUT("source pos",r_src);
            OUT("target pos",r_trg);
            OUT("source vor",a_src);
            OUT("target vor",a_trg);*/
            wrong++;
        }
    }

    std::cout << "Number wrong" << wrong << " (" << (100.0 * wrong) / iterations << " %)";
    gsl_vector_free(r_src);
    gsl_vector_free(a_src);
    gsl_vector_free(r_trg);
    gsl_vector_free(a_trg);
}

void compare_equal(gsl_vector *gpu, gsl_vector *cpu, int size, int offset) {
    for(double epsilon : epsilonsToTest) {
        int wrong = 0;
        std::cout << "Position\n";
        for (int i = 0; i < size; i++) {
            double gpu_v = gsl_vector_get(gpu, i);
            double cpu_v = gsl_vector_get(cpu, i);

            if (0 != gsl_fcmp(gpu_v, cpu_v, epsilon) && !(closeToZero(gpu_v) && closeToZero(cpu_v))) {
                /*std::cout << "Different result on gpu (" << gpu_v << ") and cpu (" << cpu_v << ") at index " << i
                          << ".\n";*/
                wrong++;
            }
        }

        if (offset != 0) {
            std::cout << "Vorticity\n";

            for (int i = offset; i < offset + size; i++) {
                double gpu_v = gsl_vector_get(gpu, i);
                double cpu_v = gsl_vector_get(cpu, i);

                if (0 != gsl_fcmp(gpu_v, cpu_v, epsilon) && !(closeToZero(gpu_v) && closeToZero(cpu_v))) {
                    /*std::cout << "Different result on gpu (" << gpu_v << ") and cpu (" << cpu_v << ") at index "
                              << i - offset << ".\n";*/
                    wrong++;
                }
            }
        }
        if (wrong == 0) {
            std::cout << "No error found with epsilon " << epsilon << "\n";
        } else {
            std::cout << wrong << " differences  (" << (100.0 * wrong) / (size * (offset == 0 ? 1 : 2)) << " %) with epsilon " << epsilon << "\n";
        }
    }
}

bool closeToZero(double gpuV) {
    return abs(gpuV) < 1e-15;
}

/*
1 differences  (2.3809523809523812e-03 %) with epsilon 1.0000000000000001e-09
2 differences  (4.7619047619047623e-03 %) with epsilon 9.9999999999999994e-12
25 differences  (5.9523809523809521e-02 %) with epsilon 9.9999999999999998e-13
378 differences  (9.0000000000000002e-01 %) with epsilon 1.0000000000000000e-13
4331 differences  (1.0311904761904762e+01 %) with epsilon 1.0000000000000000e-14
31256 differences  (7.4419047619047618e+01 %) with epsilon 1.0000000000000001e-15
 */
void singleStep(){
    unsigned long int seed1 = 53478496;
    unsigned long int seed2 = 3543753850;

    int size1 = 3000;
    int size2 = 4000;


    gsl_rng * r;
    const gsl_rng_type * T;

    gsl_rng_env_setup();
    T = gsl_rng_default;
    r = gsl_rng_alloc (T);

    gsl_rng_set(r, seed1);
    test_wake wakeGPU = test_wake(size1, r);
    gsl_rng_set(r, seed1);
    test_wake wakeCPU = test_wake(size1, r);

    gsl_rng_set(r, seed2);
    test_wake wakeGPU2 = test_wake(size2, r);
    gsl_rng_set(r, seed2);
    test_wake wakeCPU2 = test_wake(size2, r);

    __interaction *interactionGPU = new gpu<>(&wakeGPU, &wakeGPU2);
    __interaction *interactionCPU = new __parallel(&wakeCPU, &wakeCPU2);

    std::cout << "Solve GPU\n";
    interactionGPU->solve();
    std::cout << "Solve CPU\n";
    interactionCPU->solve();

    std::cout << "compare\n";
    gsl_vector *ratesGPU = gsl_vector_calloc(wakeGPU._size + wakeGPU2._size);
    gsl_vector *ratesCPU = gsl_vector_calloc(wakeCPU._size + wakeCPU2._size);

    interactionGPU->getRates(ratesGPU);
    interactionCPU->getRates(ratesCPU);

    compare_equal(ratesGPU, ratesCPU, wakeGPU._size + wakeGPU2._size, 0);
}

/*
No error found with epsilon 1.0000000000000000e-10
No error found with epsilon 9.9999999999999994e-12
7 differences  (5.9003017582899241e-03 %) with epsilon 9.9999999999999998e-13
393 differences  (3.3125979871542000e-01 %) with epsilon 1.0000000000000000e-13
6303 differences  (5.3128002832144841e+00 %) with epsilon 1.0000000000000000e-14
 */

void wholeIntegration(){
    /*
    unsigned long int seed1 = 98;

    int size1 = 3000;

    gsl_rng * r;
    const gsl_rng_type * T;

    gsl_rng_env_setup();
    T = gsl_rng_default;
    r = gsl_rng_alloc (T);

    gsl_rng_set(r, seed1);
    test_wake wakeGPU = test_wake(size1, r);
    gsl_rng_set(r, seed1);
    test_wake wakeCPU = test_wake(size1, r);
*/

    pawan::__io *IOvringGPU = new pawan::__io("testGPU");


    pawan::__wake *W = new pawan::__vring(1.0,0.1,6,117,0.0735);
    pawan::__interaction *S = new pawan::__parallel(W);

    pawan::__resolve *R = new pawan::__resolve();
    S->diagnose();//simply calculate diagnostics
    R->rebuild(S,IOvringGPU);
    W->print();
    S->diagnose();
    S->solve();
    W->print();

    //pawan::__wake *W1 = new pawan::__ring(1.0,5.0,0.1,1000);
    pawan::__wake *wakeGPU = new pawan::__wake(W);
    //pawan::__wake *W2 = new pawan::__ring(1.0,5.0,0.1,1000);
    pawan::__wake *wakeCPU = new pawan::__wake(W);


    pawan::__interaction *SvringGPU = new pawan::__parallel(wakeGPU);
    pawan::__integration *INvringGPU = new pawan::gpu_euler<>(5,100);

    INvringGPU->integrate(SvringGPU,IOvringGPU,false);

    pawan::__io *IOvringCPU = new pawan::__io("testCPU");

    pawan::__interaction *SvringCPU = new pawan::__parallel(wakeCPU);
    pawan::__integration *INvringCPU = new pawan::__integration(5,100);

    INvringCPU->integrate(SvringCPU,IOvringCPU,false);

    std::cout << "compare\n";
    gsl_vector *statesGPU = gsl_vector_calloc(wakeGPU->_maxsize);
    gsl_vector *statesCPU = gsl_vector_calloc(wakeCPU->_maxsize);

    wakeGPU->getStates(statesGPU);
    wakeCPU->getStates(statesCPU);

    compare_equal(statesGPU, statesCPU, wakeGPU->_numParticles * 3, wakeGPU->_maxsize/2);


}

void testParameters(){
    std::cout << "Setup\n";
    unsigned long int seed1 = 17478738;

    int size1 = 500000;

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
            constexpr int threadblocks = 512, unrollFactor = 1;
            std::cout << "Testing with threadsblocks: " << threadblocks << " and unrollFactor: " << unrollFactor
                      << "\n";
            gsl_rng_set(r, seed1);
            pawan::test_wake wakeGPU = pawan::test_wake(size1, r);
            pawan::__interaction *interactionGPU = new pawan::gpu<threadblocks, unrollFactor>(&wakeGPU);
            measure(interactionGPU);
            delete interactionGPU;
        }
        /*
        //Adding all of these will slow down compilation significantly!
        {
            constexpr int threadblocks = 512, unrollFactor = 64;
            std::cout << "Testing with threadsblocks: " << threadblocks << " and unrollFactor: " << unrollFactor
                      << "\n";
            gsl_rng_set(r, seed1);
            pawan::test_wake wakeGPU = pawan::test_wake(size1, r);
            pawan::__interaction *interactionGPU = new pawan::gpu<threadblocks, unrollFactor>(&wakeGPU);
            measure(interactionGPU);
            delete interactionGPU;
        }

        {
            constexpr int threadblocks = 512, unrollFactor = 128;
            std::cout << "Testing with threadsblocks: " << threadblocks << " and unrollFactor: " << unrollFactor
                      << "\n";
            gsl_rng_set(r, seed1);
            pawan::test_wake wakeGPU = pawan::test_wake(size1, r);
            pawan::__interaction *interactionGPU = new pawan::gpu<threadblocks, unrollFactor>(&wakeGPU);
            measure(interactionGPU);
            delete interactionGPU;
        }

        {
            constexpr int threadblocks = 512, unrollFactor = 256;
            std::cout << "Testing with threadsblocks: " << threadblocks << " and unrollFactor: " << unrollFactor
                      << "\n";
            gsl_rng_set(r, seed1);
            pawan::test_wake wakeGPU = pawan::test_wake(size1, r);
            pawan::__interaction *interactionGPU = new pawan::gpu<threadblocks, unrollFactor>(&wakeGPU);
            measure(interactionGPU);
            delete interactionGPU;
        }

        {
            constexpr int threadblocks = 512, unrollFactor = 512;
            std::cout << "Testing with threadsblocks: " << threadblocks << " and unrollFactor: " << unrollFactor
                      << "\n";
            gsl_rng_set(r, seed1);
            pawan::test_wake wakeGPU = pawan::test_wake(size1, r);
            pawan::__interaction *interactionGPU = new pawan::gpu<threadblocks, unrollFactor>(&wakeGPU);
            measure(interactionGPU);
            delete interactionGPU;
        }

        {
            constexpr int threadblocks = 256, unrollFactor = 1;
            std::cout << "Testing with threadsblocks: " << threadblocks << " and unrollFactor: " << unrollFactor
                      << "\n";
            gsl_rng_set(r, seed1);
            pawan::test_wake wakeGPU = pawan::test_wake(size1, r);
            pawan::__interaction *interactionGPU = new pawan::gpu<threadblocks, unrollFactor>(&wakeGPU);
            measure(interactionGPU);
            delete interactionGPU;
        }

        {
            constexpr int threadblocks = 256, unrollFactor = 32;
            std::cout << "Testing with threadsblocks: " << threadblocks << " and unrollFactor: " << unrollFactor
                      << "\n";
            gsl_rng_set(r, seed1);
            pawan::test_wake wakeGPU = pawan::test_wake(size1, r);
            pawan::__interaction *interactionGPU = new pawan::gpu<threadblocks, unrollFactor>(&wakeGPU);
            measure(interactionGPU);
            delete interactionGPU;
        }

        {
            constexpr int threadblocks = 256, unrollFactor = 64;
            std::cout << "Testing with threadsblocks: " << threadblocks << " and unrollFactor: " << unrollFactor
                      << "\n";
            gsl_rng_set(r, seed1);
            pawan::test_wake wakeGPU = pawan::test_wake(size1, r);
            pawan::__interaction *interactionGPU = new pawan::gpu<threadblocks, unrollFactor>(&wakeGPU);
            measure(interactionGPU);
            delete interactionGPU;
        }

        {
            constexpr int threadblocks = 256, unrollFactor = 128;
            std::cout << "Testing with threadsblocks: " << threadblocks << " and unrollFactor: " << unrollFactor
                      << "\n";
            gsl_rng_set(r, seed1);
            pawan::test_wake wakeGPU = pawan::test_wake(size1, r);
            pawan::__interaction *interactionGPU = new pawan::gpu<threadblocks, unrollFactor>(&wakeGPU);
            measure(interactionGPU);
            delete interactionGPU;
        }

        {
            constexpr int threadblocks = 256, unrollFactor = 256;
            std::cout << "Testing with threadsblocks: " << threadblocks << " and unrollFactor: " << unrollFactor
                      << "\n";
            gsl_rng_set(r, seed1);
            pawan::test_wake wakeGPU = pawan::test_wake(size1, r);
            pawan::__interaction *interactionGPU = new pawan::gpu<threadblocks, unrollFactor>(&wakeGPU);
            measure(interactionGPU);
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
            constexpr int threadblocks = 128, unrollFactor = 16;
            std::cout << "Testing with threadsblocks: " << threadblocks << " and unrollFactor: " << unrollFactor
                      << "\n";
            gsl_rng_set(r, seed1);
            pawan::test_wake wakeGPU = pawan::test_wake(size1, r);
            pawan::__interaction *interactionGPU = new pawan::gpu<threadblocks, unrollFactor>(&wakeGPU);
            measure(interactionGPU);
            delete interactionGPU;
        }

        {
            constexpr int threadblocks = 128, unrollFactor = 32;
            std::cout << "Testing with threadsblocks: " << threadblocks << " and unrollFactor: " << unrollFactor
                      << "\n";
            gsl_rng_set(r, seed1);
            pawan::test_wake wakeGPU = pawan::test_wake(size1, r);
            pawan::__interaction *interactionGPU = new pawan::gpu<threadblocks, unrollFactor>(&wakeGPU);
            measure(interactionGPU);
            delete interactionGPU;
        }

        {
            constexpr int threadblocks = 128, unrollFactor = 64;
            std::cout << "Testing with threadsblocks: " << threadblocks << " and unrollFactor: " << unrollFactor
                      << "\n";
            gsl_rng_set(r, seed1);
            pawan::test_wake wakeGPU = pawan::test_wake(size1, r);
            pawan::__interaction *interactionGPU = new pawan::gpu<threadblocks, unrollFactor>(&wakeGPU);
            measure(interactionGPU);
            delete interactionGPU;
        }

        {
            constexpr int threadblocks = 128, unrollFactor = 128;
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
            constexpr int threadblocks = 64, unrollFactor = 8;
            std::cout << "Testing with threadsblocks: " << threadblocks << " and unrollFactor: " << unrollFactor
                      << "\n";
            gsl_rng_set(r, seed1);
            pawan::test_wake wakeGPU = pawan::test_wake(size1, r);
            pawan::__interaction *interactionGPU = new pawan::gpu<threadblocks, unrollFactor>(&wakeGPU);
            measure(interactionGPU);
            delete interactionGPU;
        }

        {
            constexpr int threadblocks = 64, unrollFactor = 16;
            std::cout << "Testing with threadsblocks: " << threadblocks << " and unrollFactor: " << unrollFactor
                      << "\n";
            gsl_rng_set(r, seed1);
            pawan::test_wake wakeGPU = pawan::test_wake(size1, r);
            pawan::__interaction *interactionGPU = new pawan::gpu<threadblocks, unrollFactor>(&wakeGPU);
            measure(interactionGPU);
            delete interactionGPU;
        }

        {
            constexpr int threadblocks = 64, unrollFactor = 32;
            std::cout << "Testing with threadsblocks: " << threadblocks << " and unrollFactor: " << unrollFactor
                      << "\n";
            gsl_rng_set(r, seed1);
            pawan::test_wake wakeGPU = pawan::test_wake(size1, r);
            pawan::__interaction *interactionGPU = new pawan::gpu<threadblocks, unrollFactor>(&wakeGPU);
            measure(interactionGPU);
            delete interactionGPU;
        }

        {
            constexpr int threadblocks = 64, unrollFactor = 64;
            std::cout << "Testing with threadsblocks: " << threadblocks << " and unrollFactor: " << unrollFactor
                      << "\n";
            gsl_rng_set(r, seed1);
            pawan::test_wake wakeGPU = pawan::test_wake(size1, r);
            pawan::__interaction *interactionGPU = new pawan::gpu<threadblocks, unrollFactor>(&wakeGPU);
            measure(interactionGPU);
            delete interactionGPU;
        }

        {
            constexpr int threadblocks = 32, unrollFactor = 1;
            std::cout << "Testing with threadsblocks: " << threadblocks << " and unrollFactor: " << unrollFactor
                      << "\n";
            gsl_rng_set(r, seed1);
            pawan::test_wake wakeGPU = pawan::test_wake(size1, r);
            pawan::__interaction *interactionGPU = new pawan::gpu<threadblocks, unrollFactor>(&wakeGPU);
            measure(interactionGPU);
            delete interactionGPU;
        }

        {
            constexpr int threadblocks = 32, unrollFactor = 2;
            std::cout << "Testing with threadsblocks: " << threadblocks << " and unrollFactor: " << unrollFactor
                      << "\n";
            gsl_rng_set(r, seed1);
            pawan::test_wake wakeGPU = pawan::test_wake(size1, r);
            pawan::__interaction *interactionGPU = new pawan::gpu<threadblocks, unrollFactor>(&wakeGPU);
            measure(interactionGPU);
            delete interactionGPU;
        }

        {
            constexpr int threadblocks = 32, unrollFactor = 4;
            std::cout << "Testing with threadsblocks: " << threadblocks << " and unrollFactor: " << unrollFactor
                      << "\n";
            gsl_rng_set(r, seed1);
            pawan::test_wake wakeGPU = pawan::test_wake(size1, r);
            pawan::__interaction *interactionGPU = new pawan::gpu<threadblocks, unrollFactor>(&wakeGPU);
            measure(interactionGPU);
            delete interactionGPU;
        }

        {
            constexpr int threadblocks = 32, unrollFactor = 8;
            std::cout << "Testing with threadsblocks: " << threadblocks << " and unrollFactor: " << unrollFactor
                      << "\n";
            gsl_rng_set(r, seed1);
            pawan::test_wake wakeGPU = pawan::test_wake(size1, r);
            pawan::__interaction *interactionGPU = new pawan::gpu<threadblocks, unrollFactor>(&wakeGPU);
            measure(interactionGPU);
            delete interactionGPU;
        }

        {
            constexpr int threadblocks = 16, unrollFactor = 1;
            std::cout << "Testing with threadsblocks: " << threadblocks << " and unrollFactor: " << unrollFactor
                      << "\n";
            gsl_rng_set(r, seed1);
            pawan::test_wake wakeGPU = pawan::test_wake(size1, r);
            pawan::__interaction *interactionGPU = new pawan::gpu<threadblocks, unrollFactor>(&wakeGPU);
            measure(interactionGPU);
            delete interactionGPU;
        }

        {
            constexpr int threadblocks = 16, unrollFactor = 2;
            std::cout << "Testing with threadsblocks: " << threadblocks << " and unrollFactor: " << unrollFactor
                      << "\n";
            gsl_rng_set(r, seed1);
            pawan::test_wake wakeGPU = pawan::test_wake(size1, r);
            pawan::__interaction *interactionGPU = new pawan::gpu<threadblocks, unrollFactor>(&wakeGPU);
            measure(interactionGPU);
            delete interactionGPU;
        }

        {
            constexpr int threadblocks = 16, unrollFactor = 4;
            std::cout << "Testing with threadsblocks: " << threadblocks << " and unrollFactor: " << unrollFactor
                      << "\n";
            gsl_rng_set(r, seed1);
            pawan::test_wake wakeGPU = pawan::test_wake(size1, r);
            pawan::__interaction *interactionGPU = new pawan::gpu<threadblocks, unrollFactor>(&wakeGPU);
            measure(interactionGPU);
            delete interactionGPU;
        }

        {
            constexpr int threadblocks = 16, unrollFactor = 8;
            std::cout << "Testing with threadsblocks: " << threadblocks << " and unrollFactor: " << unrollFactor
                      << "\n";
            gsl_rng_set(r, seed1);
            pawan::test_wake wakeGPU = pawan::test_wake(size1, r);
            pawan::__interaction *interactionGPU = new pawan::gpu<threadblocks, unrollFactor>(&wakeGPU);
            measure(interactionGPU);
            delete interactionGPU;
        }
         */
    }
}


void test()
{
    //testInteractWithRandomValues();

    //singleStep();

    wholeIntegration();

}


