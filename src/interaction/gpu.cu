//GPU version for interaction
/*
 * memory layout for particle i:
 * particle[2 * i] x,y,z : position
 * particle[2 * i] w : smoothing radius
 * particle[2 * i + 1] x,y,z : vorticity
 * particle[2 * i + 1] w : volume
 *
 * rates[2 * i] : velocity
 * rates[2 * i + 1] : change in vorticity
 */

#include "gpu.cuh"
#include "interaction_utils_gpu.cuh"
#include <iostream>

namespace pawan {

    constexpr size_t threadBlockSize = 64;

    //Helper for checking errors from CUDA API
#define checkGPUError(ans) checkGPUError_((ans), __FILE__, __LINE__)
    __inline__ void checkGPUError_(cudaError_t errorCode, const char* file, int line){

        if(errorCode != cudaSuccess) {
            //report error and stop
            std::cout << "Cuda Error: " << cudaGetErrorString(errorCode) << "\nin " << file << ", line " << line;
            exit(EXIT_FAILURE);
        }
    }


    /*
     * GPU Kernel that computes the interaction of every particle with every other particle
     * Each thread is responsible for the interactions of a single particle with every other one
     */
    __global__ void interactKernel(double4 *particles, double3 *rates, const size_t N, const double nu) {

        __shared__ double4 sharedParticles[2 * threadBlockSize];

        double4 ownPosition, ownVorticity;
        double3 ownVelocity = {0,0,0}, ownRetVorticity = {0,0,0};

        size_t ownParticleIndex = blockIdx.x * threadBlockSize + threadIdx.x;

        //cache own particle if index in bounds
        if(ownParticleIndex < N){
            ownPosition = particles[2 * ownParticleIndex];
            ownVorticity = particles[2 * ownParticleIndex + 1];
        }

        //complete tiles
        size_t fullIters = N/threadBlockSize;
        size_t loadPos = threadIdx.x;
        for(int i = 0; i < fullIters; i++, loadPos += threadBlockSize){
            //load particle into shared memory to be reused by the whole threadblock
            sharedParticles[2 * threadIdx.x] = particles[2 * loadPos];
            sharedParticles[2 * threadIdx.x + 1] = particles[2 * loadPos + 1];

            __syncthreads();

            //calculate interaction with every particle in the shared memory
            //skip if own particle out of bounds
            if(ownParticleIndex < N) {
                for(int j = 0; j < threadBlockSize; j++){  //TODO unroll?

                    if(! (blockIdx.x == i && threadIdx.x == j) ) //skip interaction with yourself, as the interact function can not handle it
                        INTERACT_GPU(nu, ownPosition, sharedParticles[2*j], ownVorticity,
                                     sharedParticles[2*j + 1], ownVelocity, ownRetVorticity);
                }
            }

            __syncthreads();
        }

        //last tile may be incomplete
        //do not load out of bound particles
        if(loadPos < N) {
            sharedParticles[2 * threadIdx.x] = particles[2 * loadPos];
            sharedParticles[2 * threadIdx.x + 1] = particles[2 * loadPos + 1];
        }
        __syncthreads();

        if(ownParticleIndex < N) {
            //finish incomplete tile
            size_t remainingParticles = N % threadBlockSize;
            for (int j = 0; j < remainingParticles; j++) { //TODO unroll?

                if(! (blockIdx.x == fullIters && threadIdx.x == j) ) //skip interaction with yourself, as the interact function can not handle it
                    INTERACT_GPU(nu, ownPosition, sharedParticles[2*j], ownVorticity,
                                 sharedParticles[2*j + 1], ownVelocity, ownRetVorticity);
            }

            //write results to global memory
            rates[2 * ownParticleIndex] = ownVelocity;
            rates[2 * ownParticleIndex + 1] = ownRetVorticity;

        }

        //no sync needed because no further writes take place

    }

    pawan::gpu::gpu(__wake *W):__parallel(W){}
    pawan::gpu::gpu(__wake *W1, __wake *W2):__parallel(W1,W2){}

    double4 *copyParticlesToGPU(size_t size, std::vector<pawan::__wake *> &W);
    void copyRatesFromGPU(double3 *ratesGPU, size_t size, const std::vector<pawan::__wake *>& W);

    void gpu::interact() {

        size_t totalParticles = 0;

        for(auto w : _W ){
            totalParticles += w->_numParticles;
        }

        double4 * particlesGPU = copyParticlesToGPU(totalParticles,_W);

        //allocate result array on GPU
        //TODO: would double4 bring performance improvements? Maybe only if kept in memory
        double3 * ratesGPU;
        checkGPUError(cudaMalloc(&ratesGPU, totalParticles * 2 * sizeof(double3)));

        //call kernel
        size_t threadBlocks = (totalParticles + threadBlockSize - 1) / threadBlockSize; //round up!
        interactKernel<<<threadBlocks, threadBlockSize>>>(particlesGPU, ratesGPU, totalParticles, _nu);

        copyRatesFromGPU(ratesGPU, totalParticles, _W);

        //free memory on GPU
        //TODO: keep particles on GPU the whole time
        checkGPUError(cudaFree(ratesGPU));
        checkGPUError(cudaFree(particlesGPU));
    }

    /*
     * Copy the particle information from the wakes to the GPU
     */
    double4* copyParticlesToGPU(size_t size, std::vector<pawan::__wake *> &W) {

        size_t mem_size = size * 2 * sizeof(double4);

        double4 *particlesGPU, *particlesBuffer;

        //use this instead of malloc to have pinned memory and therefore a faster copy
        //TODO flags at initialisation? -> documentation
        checkGPUError(cudaHostAlloc(&particlesBuffer, mem_size,cudaHostAllocWriteCombined));
        checkGPUError(cudaMalloc(&particlesGPU, mem_size));

        //fill buffer with data
        int position = 0;

        for(auto const w : W){
            for(int i = 0; i < w->_numParticles; i++, position++) {
                particlesBuffer[2 * position].x = gsl_matrix_get(w->_position, i, 0);
                particlesBuffer[2 * position].y = gsl_matrix_get(w->_position, i, 1);
                particlesBuffer[2 * position].z = gsl_matrix_get(w->_position, i, 2);
                particlesBuffer[2 * position].w = gsl_vector_get(w->_radius, i);
                particlesBuffer[2 * position + 1].x = gsl_matrix_get(w->_vorticity, i, 0);
                particlesBuffer[2 * position + 1].y = gsl_matrix_get(w->_vorticity, i, 1);
                particlesBuffer[2 * position + 1].z = gsl_matrix_get(w->_vorticity, i, 2);
                particlesBuffer[2 * position + 1].w = gsl_vector_get(w->_volume, i);

            }
        }


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

} // pawan