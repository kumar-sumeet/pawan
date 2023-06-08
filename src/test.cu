
#include "test.cuh"
#include <iostream>

#include <gsl/gsl_vector_double.h>
#include "interaction/interaction_utils.h"
#include "interaction/interaction_utils_gpu.cuh"
#include <gsl/gsl_rng.h>


//output errors with location
#define checkGPUError(ans) checkGPUError_((ans), __FILE__, __LINE__)

__inline__ void checkGPUError_(cudaError_t errorCode, const char* file, int line){

    if(errorCode != cudaSuccess) {
        //report error and stop
        std::cout << "Cuda Error: " << cudaGetErrorString(errorCode) << "\nin " << file << ", line " << line;
        exit(EXIT_FAILURE);
    }
}

__global__ void testKernel(double nu,
                           const double4 *data,
                           double3 *returnVals){
    INTERACT_GPU(nu, data[0], data[2], data[1], data[3], returnVals[0], returnVals[1]);
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

    constexpr double epsilon = 1e-13;

    bool equal = gsl_fcmp(retVals[0].x, vx_s, epsilon) == 0
                 && gsl_fcmp(retVals[0].y, vy_s, epsilon) == 0
                 && gsl_fcmp(retVals[0].z, vz_s, epsilon) == 0
                 && gsl_fcmp(retVals[1].x, qx_s, epsilon) == 0
                 && gsl_fcmp(retVals[1].y, qy_s, epsilon) == 0
                 && gsl_fcmp(retVals[1].z, qz_s, epsilon) == 0;

    checkGPUError(cudaFree(data));
    checkGPUError(cudaFree(retVals));
    gsl_vector_free(dr_trg);
    gsl_vector_free(da_trg);

    return equal;
}


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
            std::cout << "Number "<< i << " is wrong! nu:" << nu << " s: " << s_src << ", " << s_trg << " v: " <<v_src << ", " << v_trg << "\n";
            OUT("source pos",r_src);
            OUT("target pos",r_trg);
            OUT("source vor",a_src);
            OUT("target vor",a_trg);
            wrong++;
        }
    }

    std::cout << "Number wrong" << wrong << " (" << (100.0 * wrong) / iterations << " %)";
    gsl_vector_free(r_src);
    gsl_vector_free(a_src);
    gsl_vector_free(r_trg);
    gsl_vector_free(a_trg);
}

void test()
{
    testInteractWithRandomValues();


}


