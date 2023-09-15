
#pragma once

#include "interaction/interaction_utils_gpu.cuh"
#include "wake/wake.h"
#include "vector"
#include "system/system.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace cg = cooperative_groups;

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

//Helper for checking errors from CUDA API
#define checkGPUError(ans) checkGPUError_((ans), __FILE__, __LINE__)
__inline__ void checkGPUError_(cudaError_t errorCode, const char* file, int line){

    if(errorCode != cudaSuccess) {
        //report error and stop
        std::cout << "Cuda Error: " << cudaGetErrorString(errorCode) << "\nin " << file << ", line " << line << "\n";
        exit(EXIT_FAILURE);
    }
}

double4 *copyParticlesToGPU(size_t size, pawan::__system *s);
void copyRatesFromGPU(double3 *ratesGPU, size_t size, const std::vector<pawan::__wake *>& W);

/*!
 * Computes the interaction of a single particle with every other one
 * @param particles data for all particles
 * @param N number of particles
 * @param nu
 * @param position position of this specific particle
 * @param vorticity vorticity of this specific particle
 * @param index number of this particle (not index in particle array!)
 * @param velocity reference to store resulting rate
 * @param retVorticity reference to store resulting rate
 */
 template<int threadBlockSize = 128, int unrollFactor = 1>
__device__ inline void interact_with_all(const double4 *particles, const size_t N, const double nu, const double4 &position,
                                         const double4 &vorticity, const size_t index, double3 &velocity,
                                         double3 &retVorticity) {
    __shared__ double4 sharedParticles[2 * threadBlockSize];

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
        if(index < N) {
            #pragma unroll unrollFactor
            for(int j = 0; j < threadBlockSize; j++){

                if(! (blockIdx.x == i && threadIdx.x == j) ) //skip interaction with yourself, as the interact function can not handle it
                    INTERACT_GPU(nu, position, sharedParticles[2 * j], vorticity,
                                 sharedParticles[2*j + 1], velocity, retVorticity);
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

    if(index < N) {
        //finish incomplete tile
        size_t remainingParticles = N % threadBlockSize;
        #pragma unroll unrollFactor
        for (int j = 0; j < remainingParticles; j++) {

            if(! (blockIdx.x == fullIters && threadIdx.x == j) ) //skip interaction with yourself, as the interact function can not handle it
                INTERACT_GPU(nu, position, sharedParticles[2 * j], vorticity,
                             sharedParticles[2*j + 1], velocity, retVorticity);
        }
    }
}

template<int threadBlockSize = 128, int unrollFactor = 1>
__device__ inline void reduction_lindiag(const double4 *particles, const size_t numparticles,
                                         double *totalSum, const int op) {
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
    extern __shared__ double3 reddata[];

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    double3 mySum={0.0,0.0,0.0};
    double3 temp={0.0,0.0,0.0};
    if(i < numparticles) {
        switch(op){
            case 0: //totalvorticity
                mySum = getvorticity(particles,i);
                break;
            case 1://linearimpulse
                mySum = getlinearimpulse(particles,i);
                break;
            case 2://angularimpulse
                mySum = getangularimpulse(particles,i);
                break;
            case 3://centroid position
                mySum = getZc(particles,i);
                break;
        }
    }

    if (i + blockDim.x < numparticles){
        switch(op) {
            case 0: //totalvorticity
                temp = getvorticity(particles, i + blockDim.x);
                break;
            case 1://linearimpulse
                temp = getlinearimpulse(particles, i + blockDim.x);
                break;
            case 2://angularimpulse
                temp = getangularimpulse(particles, i + blockDim.x);
                break;
            case 3://centroid position
                temp = getZc(particles,i);
                break;
        }
        mySum.x += temp.x;
        mySum.y += temp.y;
        mySum.z += temp.z;
    }

    reddata[tid] = mySum;
    cg::sync(cta);

    // do reduction in shared mem
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            reddata[tid].x = mySum.x = mySum.x + reddata[tid + s].x;
            reddata[tid].y = mySum.y = mySum.y + reddata[tid + s].y;
            reddata[tid].z = mySum.z = mySum.z + reddata[tid + s].z;
        }

        cg::sync(cta);
    }

    // write result for this block to global mem
    if (tid == 0){
        atomicAdd(totalSum,mySum.x);
        atomicAdd(totalSum+1,mySum.y);
        atomicAdd(totalSum+2,mySum.z);
    }
}

template<int threadBlockSize = 128, int unrollFactor = 1>
__device__ inline void quaddiag(const double4 *particles, const size_t N,
                                const double4 &position, const double4 &vorticity,
                                const size_t index, double &partDiagContrib, const int &op) {
    __shared__ double4 sharedParticles[2 * threadBlockSize];

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
        if(index < N) {
            #pragma unroll unrollFactor
            for(int j = 0; j < threadBlockSize; j++){

                switch(op) {
                    case 0: //enstrophy
                        ENSTROPHY(position, sharedParticles[2 * j], vorticity,
                                  sharedParticles[2*j + 1], partDiagContrib);
                        break;
                    case 1://kineticenergy
                        KINETICENERGY(position, sharedParticles[2 * j], vorticity,
                                      sharedParticles[2*j + 1], partDiagContrib);
                        break;
                    case 2://helicity
                        if(! (blockIdx.x == i && threadIdx.x == j) )
                            HELICITY(position, sharedParticles[2 * j], vorticity,
                                     sharedParticles[2*j + 1], partDiagContrib);
                        break;
                    case 3://enstrophyF
                        ENSTROPHYF(position, sharedParticles[2 * j], vorticity,
                                   sharedParticles[2*j + 1], partDiagContrib);
                        break;
                    case 4://kineticenergyF
                        KINETICENERGYF(position, sharedParticles[2 * j], vorticity,
                                       sharedParticles[2*j + 1], partDiagContrib);
                        break;
                }
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

    if(index < N) {
        //finish incomplete tile
        size_t remainingParticles = N % threadBlockSize;
        #pragma unroll unrollFactor
        for (int j = 0; j < remainingParticles; j++) {

            switch(op) {
                case 0: //enstrophy
                    ENSTROPHY(position, sharedParticles[2 * j], vorticity,
                              sharedParticles[2*j + 1], partDiagContrib);
                    break;
                case 1://kineticenergy
                    KINETICENERGY(position, sharedParticles[2 * j], vorticity,
                                  sharedParticles[2*j + 1], partDiagContrib);
                    break;
                case 2://helicity
                    if(! (blockIdx.x == fullIters && threadIdx.x == j) )
                        HELICITY(position, sharedParticles[2 * j], vorticity,
                                 sharedParticles[2*j + 1], partDiagContrib);
                    break;
                case 3://enstrophyF
                    ENSTROPHYF(position, sharedParticles[2 * j], vorticity,
                               sharedParticles[2*j + 1], partDiagContrib);
                    break;
                case 4://kineticenergyF
                    KINETICENERGYF(position, sharedParticles[2 * j], vorticity,
                                   sharedParticles[2*j + 1], partDiagContrib);
                    break;
            }
        }
    }
}
