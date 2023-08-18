#include "gpu_common.cuh"
#include "system/system.h"

/*
 * Copy the particle information from the wakes to the GPU
 * allocates a new array of fitting size on the gpu
 */
double4* copyParticlesToGPU(size_t size, pawan::__system *s) {

    size_t mem_size = size * 2 * sizeof(double4);

    double4 *particlesGPU, *particlesBuffer;

    //use this instead of malloc to have pinned memory and therefore a faster copy
    //TODO flags at initialisation? -> documentation
    checkGPUError(cudaHostAlloc(&particlesBuffer, mem_size,cudaHostAllocWriteCombined)); //We only write to this buffer -> use WriteCombine
    checkGPUError(cudaMalloc(&particlesGPU, mem_size));

    s->getParticles(reinterpret_cast<double*>(particlesBuffer));

    //copy to GPU and free buffer
    checkGPUError(cudaMemcpy(particlesGPU,particlesBuffer,mem_size,cudaMemcpyHostToDevice));
    checkGPUError(cudaFreeHost(particlesBuffer));

    return particlesGPU;
}

void copyRatesFromGPU(double3 *ratesGPU, size_t size, const std::vector<pawan::__wake *>& W) {

    size_t mem_size = size * 2 * sizeof(double3);

    double3 * rateBuffer;

    checkGPUError(cudaMallocHost(&rateBuffer,mem_size));
    checkGPUError(cudaMemcpy(rateBuffer,ratesGPU, mem_size, cudaMemcpyDeviceToHost));

    int position = 0;

    for(auto w : W){
        for(int i = 0; i < w->_numParticles; i++, position++){
            gsl_matrix_set(w->_velocity,i,0,rateBuffer[2*position].x);
            gsl_matrix_set(w->_velocity,i,1,rateBuffer[2*position].y);
            gsl_matrix_set(w->_velocity,i,2,rateBuffer[2*position].z);
            gsl_matrix_set(w->_retvorcity,i,0,rateBuffer[2*position+1].x);
            gsl_matrix_set(w->_retvorcity,i,1,rateBuffer[2*position+1].y);
            gsl_matrix_set(w->_retvorcity,i,2,rateBuffer[2*position+1].z);

        }
    }

    checkGPUError(cudaFreeHost(rateBuffer));
}