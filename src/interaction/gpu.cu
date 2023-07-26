#include "gpu.cuh"
#include <iostream>
#include "gpu_common.cuh"

namespace pawan {
    /*
     * GPU Kernel wrapper for interact
     * Each thread is responsible for the interactions of a single particle with every other one
     */
    __global__ void interactKernel(double4 *particles, double3 *rates, const size_t N, const double nu) {

        double4 ownPosition, ownVorticity;
        double3 ownVelocity = {0,0,0}, ownRetVorticity = {0,0,0};

        size_t index = blockIdx.x * threadBlockSize + threadIdx.x;

        //cache own particle if index in bounds
        if(index < N){
            ownPosition = particles[2 * index];
            ownVorticity = particles[2 * index + 1];
        }

        interact_with_all(particles, N, nu, ownPosition, ownVorticity, index, ownVelocity, ownRetVorticity);

        if(index < N) {
            //write results to global memory
            rates[2 * index] = ownVelocity;
            rates[2 * index + 1] = ownRetVorticity;

        }

    }

    pawan::gpu::gpu(__wake *W):__parallel(W){}
    pawan::gpu::gpu(__wake *W1, __wake *W2):__parallel(W1,W2){}


    void gpu::interact() {

        size_t totalParticles = 0;

        for(auto w : _W ){
            totalParticles += w->_numParticles;
        }

        double4 * particlesGPU = copyParticlesToGPU(totalParticles,this);

        //allocate result array on GPU
        //TODO: would double4 bring performance improvements? Maybe only if kept in memory
        double3 * ratesGPU;
        checkGPUError(cudaMalloc(&ratesGPU, totalParticles * 2 * sizeof(double3)));

        //call kernel
        size_t threadBlocks = (totalParticles + threadBlockSize - 1) / threadBlockSize; //round up!
        interactKernel<<<threadBlocks, threadBlockSize>>>(particlesGPU, ratesGPU, totalParticles, _nu);

        copyRatesFromGPU(ratesGPU, totalParticles, _W);

        //free memory on GPU
        checkGPUError(cudaFree(ratesGPU));
        checkGPUError(cudaFree(particlesGPU));
    }

} // pawan