
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
__device__ inline void interact_with_all(const double4 *particles, int source_age, const size_t N, const double nu, const double4 &position,
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
                    INTERACT_GPU(nu, source_age, position, sharedParticles[2 * j], vorticity,
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
                INTERACT_GPU(nu, source_age, position, sharedParticles[2 * j], vorticity,
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
            case 4://induced velocity at origin
                double3 origin {0,0,0};
                mySum = getVi(particles[2*i],particles[2*i+1],origin);
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
                temp = getZc(particles,i + blockDim.x);
                break;
            case 4://induced velocity at origin
                double3 origin {0,0,0};
                temp = getVi(particles[2*(i + blockDim.x)],particles[2*(i + blockDim.x)+1],origin);
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

template<int threadBlockSize = 128, int unrollFactor = 1>
__device__ inline void divfreeomega_contrib_all(const double4 *particles, const size_t N,
                                              const double4 &position, const size_t index,
                                              double3 &divfreeomega_p) {
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
        if(index < N) {
#pragma unroll unrollFactor
            for(int j = 0; j < threadBlockSize; j++){

                if(! (blockIdx.x == i && threadIdx.x == j) ) //skip self-interaction
                    DIVFREEOMEGA_GPU(position, sharedParticles[2 * j], sharedParticles[2 * j + 1], divfreeomega_p);
            }
        }

        __syncthreads();
    }

    //separate treatment of last tile since it may be incomplete
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

            if(! (blockIdx.x == fullIters && threadIdx.x == j) ) //skip self-interaction
                DIVFREEOMEGA_GPU(position, sharedParticles[2 * j], sharedParticles[2 * j + 1], divfreeomega_p);
        }
    }
}

template<int threadBlockSize = 128, int unrollFactor = 1>
__device__ inline void gridSol_contrib_all(const double4 *particles,
                                           const size_t Npart,
                                           const double3 &nodePos,
                                           double3 &nodeVel,
                                           double3 &nodeVor) {
    __shared__ double4 sharedParticles[2 * threadBlockSize];

    size_t fullIters = Npart/threadBlockSize;
    size_t partLoadPos = threadIdx.x;

    for(int i = 0; i < fullIters; i++, partLoadPos += threadBlockSize){
        sharedParticles[2 * threadIdx.x] = particles[2 * partLoadPos];
        sharedParticles[2 * threadIdx.x + 1] = particles[2 * partLoadPos + 1];

        __syncthreads();

        #pragma unroll unrollFactor
        for (int j = 0; j < threadBlockSize; j++) {
            GRIDSOL_GPU(nodePos, sharedParticles[2 * j], sharedParticles[2 * j + 1], nodeVel, nodeVor);
        }
        __syncthreads();
    }

    if(partLoadPos < Npart) {
        sharedParticles[2 * threadIdx.x] = particles[2 * partLoadPos];
        sharedParticles[2 * threadIdx.x + 1] = particles[2 * partLoadPos + 1];
    }

    __syncthreads();

    size_t remainingParticles = Npart % threadBlockSize;
    #pragma unroll unrollFactor
    for (int j = 0; j < remainingParticles; j++) {
        GRIDSOL_GPU(nodePos, sharedParticles[2 * j], sharedParticles[2 * j + 1], nodeVel, nodeVor);
    }
}
/*template<int threadBlockSize = 128, int unrollFactor = 1>
__device__ inline void boundVorVind_contrib_all(const double3 *airStaPos,
                                        const double *circ,
                                        const size_t NairSta,
                                        double4 partposition) {
    __shared__ double4 sharedAirstations[threadBlockSize];

    size_t fullIters = NairSta/threadBlockSize;
    size_t airstaLoadPos = threadIdx.x;

    for(int i = 0; i < fullIters; i++, airstaLoadPos += threadBlockSize){
        sharedAirstations[threadIdx.x].x = airStaPos[airstaLoadPos].x;
        sharedAirstations[threadIdx.x].y = airStaPos[airstaLoadPos].y;
        sharedAirstations[threadIdx.x].z = airStaPos[airstaLoadPos].z;
        sharedAirstations[threadIdx.x].w = circ[airstaLoadPos];

        __syncthreads();

        #pragma unroll unrollFactor
        for (int j = 0; j < threadBlockSize; j++) {
            BOUNDVORVIND_GPU(nodePos, sharedAirstations[j], sharedAirstations[j + 1], nodeVel, nodeVor);
        }
        __syncthreads();
    }

    if(airstaLoadPos < NairSta) {
        sharedAirstations[threadIdx.x] = particles[airstaLoadPos];
        sharedAirstations[threadIdx.x + 1] = particles[airstaLoadPos + 1];
    }

    __syncthreads();

    size_t remainingParticles = Npart % threadBlockSize;
    #pragma unroll unrollFactor
    for (int j = 0; j < remainingParticles; j++) {
        BOUNDVORVIND_GPU(nodePos, sharedAirstations[j], sharedAirstations[j + 1], nodeVel, nodeVor);
    }
}*/

template<int threadBlockSize = 128, int unrollFactor = 1>
__device__ inline void inflow_contrib_all(const double4 *particles,
                                           const size_t Npart,
                                           const double3 &airStaPos,
                                           double3 &airStaVel) {
    __shared__ double4 sharedParticles[2 * threadBlockSize];

    size_t fullIters = Npart/threadBlockSize;
    size_t partLoadPos = threadIdx.x;

    for(int i = 0; i < fullIters; i++, partLoadPos += threadBlockSize){
        sharedParticles[2 * threadIdx.x] = particles[2 * partLoadPos];
        sharedParticles[2 * threadIdx.x + 1] = particles[2 * partLoadPos + 1];
        __syncthreads();

        #pragma unroll unrollFactor
        for (int j = 0; j < threadBlockSize; j++) {
            getVi(sharedParticles[2 * j], sharedParticles[2 * j + 1], airStaPos, airStaVel);
        }
        __syncthreads();
    }

    if(partLoadPos < Npart) {
        sharedParticles[2 * threadIdx.x] = particles[2 * partLoadPos];
        sharedParticles[2 * threadIdx.x + 1] = particles[2 * partLoadPos + 1];
    }

    __syncthreads();

    size_t remainingParticles = Npart % threadBlockSize;
    #pragma unroll unrollFactor
    for (int j = 0; j < remainingParticles; j++) {
        getVi(sharedParticles[2 * j], sharedParticles[2 * j + 1], airStaPos, airStaVel);
    }
    printf("tid = %d , airstapos = %+10.5e, %+10.5e, %+10.5e----->  lambda = %+10.5e, %+10.5e, %+10.5e \n",
           blockIdx.x * blockDim.x + threadIdx.x,airStaPos.x,airStaPos.y,airStaPos.z,
           airStaVel.x,airStaVel.y,airStaVel.z);

}
template<int threadBlockSize = 128, int unrollFactor = 1>
__device__ inline void inflow_red(const double4 *particles,
                                  const size_t numparticles,
                                  const double3 airStaPos,
                                  double *lambda_airsta) {

    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
    extern __shared__ double3 reddata[];

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    double3 mySum={0.0,0.0,0.0};
    double3 temp={0.0,0.0,0.0};
    if(i < numparticles)
        mySum = getVi(particles[2*i],particles[2*i+1],airStaPos);

    if (i + blockDim.x < numparticles){
        temp = getVi(particles[2*(i + blockDim.x)],particles[2*(i + blockDim.x)+1],airStaPos);
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
        atomicAdd(lambda_airsta,mySum.x);
        atomicAdd(lambda_airsta+1,mySum.y);
        atomicAdd(lambda_airsta+2,mySum.z);
    }

}
