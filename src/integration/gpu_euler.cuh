#pragma once

#include "interaction/gpu_common.cuh"
#include "integration.h"
#include "interaction/gpu.cuh"
#include "wake/wake.h"


namespace pawan{
    template<int threadBlockSize = 128, int unrollFactor = 1>
    class gpu_euler : public __integration{

    public:
        gpu_euler(const double &t, const size_t &n);
        gpu_euler();

        ~gpu_euler() = default;

        void integrate(__system *S,
                       __io *IO,
                       NetworkInterfaceTCP<OPawanRecvData,OPawanSendData> *networkCommunicatorTest,
                       bool diagnose=false) override;

        void integrate(__system *S,
                       __io *IO,
                       bool diagnose=false
        ) override;
    };

}

void resizeToFit(double4 *cpu, double4 *gpu1, double4 *gpu2, size_t &size, int particles);

template<int threadBlockSize, int unrollFactor>
pawan::gpu_euler<threadBlockSize,unrollFactor>::gpu_euler(const double &t, const size_t &n):__integration(t,n){}
template<int threadBlockSize, int unrollFactor>
pawan::gpu_euler<threadBlockSize,unrollFactor>::gpu_euler():__integration(){}

template<int threadBlockSize = 128, int unrollFactor = 1>
__global__ void eulerKernel(const double4 *source, double4 *target, const size_t N, const double nu, double dt) {

    double4 ownPosition, ownVorticity;
    double3 ownVelocity = {0,0,0}, ownRetVorticity = {0,0,0};

    size_t index = blockIdx.x * threadBlockSize + threadIdx.x;

    //cache own particle if index in bounds
    if(index < N){
        ownPosition = source[2 * index];
        ownVorticity = source[2 * index + 1];
    }

    interact_with_all<threadBlockSize,unrollFactor>(source, N, nu, ownPosition, ownVorticity, index, ownVelocity, ownRetVorticity);

    if(index < N) {
        //do the integration step and write result to target
        scale(dt, ownVelocity);
        add(ownPosition,ownVelocity);
        scale(dt,ownRetVorticity);
        add(ownVorticity, ownRetVorticity);

        target[2 * index] = ownPosition;
        target[2 * index + 1] = ownVorticity;

    }

}

template<int threadBlockSize = 128, int unrollFactor = 1>
__global__ void totalLinDiag(const double4 *source, const size_t N, double *diagnosticVal, const int op) {
    reduction_lindiag<threadBlockSize,unrollFactor>(source, N, diagnosticVal, op);
}
template<int threadBlockSize = 128, int unrollFactor = 1>
__global__ void totalQuadDiag(const double4 *source, const size_t N, double *diagnosticVal, const int op) {
    double4 ownPosition, ownVorticity;
    double partDiagContrib = 0.0;

    size_t index = blockIdx.x * threadBlockSize + threadIdx.x;

    //cache own particle if index in bounds
    if(index < N){
        ownPosition = source[2 * index];
        ownVorticity = source[2 * index + 1];
    }

    quaddiag<threadBlockSize,unrollFactor>(source, N, ownPosition, ownVorticity, index, partDiagContrib, op);
    //reduction
    atomicAdd(diagnosticVal, partDiagContrib);
}

template<int threadBlockSize, int unrollFactor>
void pawan::gpu_euler<threadBlockSize,unrollFactor>::integrate(pawan::__system *S, pawan::__io *IO,
                                                               NetworkInterfaceTCP<OPawanRecvData, OPawanSendData> *networkCommunicatorTest,
                                                               bool diagnose) {
    //Because openmp does not work in cuda files currently, we switch measurement system
    auto tStart = std::chrono::high_resolution_clock::now();
    FILE *fdiag = IO->create_binary_file(".diagnosis");
    OPawanRecvData opawanrecvdata;
    networkCommunicatorTest->getrecieveBuffer(opawanrecvdata);
    _t = opawanrecvdata.t;
    FILE *f = IO->create_binary_file(".wake");
    if(diagnose) {
        S->writenu(fdiag);
        S->diagnose();
        fwrite(&_t,sizeof(double),1,fdiag);
        S->writediagnosis(fdiag);
    }

    //Memory allocations:
    //two GPU buffers so that in each step the result can be written
    //without having to wait on all threads finishing
    //one pinned memory buffer on the cpu for copying states back
    int numberOfParticles = S->amountParticles();
    int maxnumberOfParticles = S->totalmaxParticles();
    size_t mem_size = maxnumberOfParticles * 2 * sizeof(double4);

    double4 *gpuSource, *gpuTarget, *cpuBuffer;
    checkGPUError(cudaMallocHost(&cpuBuffer, mem_size));
    checkGPUError(cudaMalloc(&gpuSource, mem_size));
    checkGPUError(cudaMalloc(&gpuTarget, mem_size));

    //Transfer particles to GPU
    S->getParticles(reinterpret_cast<double *>(cpuBuffer));
    checkGPUError(cudaMemcpy(gpuSource,cpuBuffer,mem_size,cudaMemcpyHostToDevice));


    size_t stepnum = 0;
    while(_t <= opawanrecvdata.tfinal){
        OUT("\tTime",_t);
        OUT("\tStepNum",stepnum);

        //number of particles might have changed
        numberOfParticles = S->amountParticles();
        //mem_size = numberOfParticles * 2 * sizeof(double4);

        //resizeToFit(cpuBuffer, gpuSource, gpuTarget, mem_size, S->amountParticles());
        S->getParticles(reinterpret_cast<double *>(cpuBuffer));
        checkGPUError(cudaMemcpy(gpuSource,cpuBuffer,mem_size,cudaMemcpyHostToDevice));

        size_t threadBlocks = (numberOfParticles + threadBlockSize - 1) / threadBlockSize;
        eulerKernel<threadBlockSize, unrollFactor><<<threadBlocks, threadBlockSize>>>(gpuSource,gpuTarget,numberOfParticles,S->getNu(), _dt);

        checkGPUError(cudaMemcpy(cpuBuffer,gpuTarget,mem_size,cudaMemcpyDeviceToHost));
        S->setParticles(reinterpret_cast<double *>(cpuBuffer));

        //S->relax(stepnum);
        if(diagnose){
            S->diagnose();
            fwrite(&_t,sizeof(double),1,fdiag);
            S->writediagnosis(fdiag);
        }
        int transient_steps = 360;
        if (stepnum < transient_steps) {
            printf("Vinf = %3.2e, %3.2e, %3.2e \n", opawanrecvdata.Vinf[0], opawanrecvdata.Vinf[1], opawanrecvdata.Vinf[2]);
            opawanrecvdata.Vinf[2] = opawanrecvdata.Vinf[2] + 15 * (transient_steps - stepnum) / transient_steps;
            printf("Vinf + suppress = %3.2e, %3.2e, %3.2e", opawanrecvdata.Vinf[0], opawanrecvdata.Vinf[1],
                   opawanrecvdata.Vinf[2]);
        }


        S->updateVinfEffect(opawanrecvdata.Vinf,opawanrecvdata.deltat);
        //S->updateBoundVorEffect(&opawanrecvdata,_dt);
        fwrite(&_t,sizeof(double),1,f);
        S->write(f);  //write particles info after interaction in this time step

        OPawanSendData opawansenddata;//create it once outside the loop and should be good
        S->getInflow(&opawanrecvdata,&opawansenddata);
        networkCommunicatorTest->send_data(opawansenddata);

        //S->diagnose();
        stepnum = stepnum+1;
        if(_t <= (opawanrecvdata.tfinal - 1*opawanrecvdata.deltat)){ //run till end of dymore sim
            networkCommunicatorTest->recieve_data(opawanrecvdata);
            S->addParticles(&opawanrecvdata, stepnum);
            _t = opawanrecvdata.t;
        }
        else
            break;
    }
    fclose(f);
    auto tEnd = std::chrono::high_resolution_clock::now();

    OUT("Total Time (s)",std::chrono::duration<double>(tEnd - tStart).count());

    checkGPUError(cudaFree(gpuSource));
    checkGPUError(cudaFree(gpuTarget));
    checkGPUError(cudaFreeHost(cpuBuffer));
}
template<int threadBlockSize, int unrollFactor>
void pawan::gpu_euler<threadBlockSize,unrollFactor>::integrate(pawan::__system *S, pawan::__io *IO, bool diagnose) {
    FILE *f = IO->create_binary_file(".wake");
    FILE *fdiag = IO->create_binary_file(".diagnosis");
    double t = 0.0;
    fwrite(&t,sizeof(double),1,f);
    S->write(f);  //write particles info as is
/*    if(diagnose) {
        S->writenu(fdiag);
        fwrite(&t,sizeof(double),1,fdiag);
        S->diagnose();
        S->writediagnosis(fdiag);
    }
*/
    //Create two cuda streams so that integration and memory copies can happen at the same time
    cudaStream_t memoryStream, integrateStream;
    cudaStreamCreate(&memoryStream);
    cudaStreamCreate(&integrateStream);

    //Memory allocations:
    //two GPU buffers so that in each step the result can be written
    //without having to wait on all threads finishing
    //one pinned memory buffer on the cpu for copying states back
    int numberOfParticles = S->amountParticles();
    int maxnumberOfParticles = S->totalmaxParticles();
    size_t mem_size = maxnumberOfParticles * 2 * sizeof(double4);

    double4 *gpuSource, *gpuTarget, *cpuBuffer;
    checkGPUError(cudaMallocHost(&cpuBuffer, mem_size));
    checkGPUError(cudaMalloc(&gpuSource, mem_size));
    checkGPUError(cudaMalloc(&gpuTarget, mem_size));

    double *totalDiag;
    checkGPUError(cudaMallocHost(&totalDiag, 17*sizeof(double)));
/*    double *totalVorticity, *linearImpulse, *angularImpulse;
    double *enstrophy, *kineticenergy, *helicity, *enstrophyF, *kineticenergyF;
    checkGPUError(cudaMallocHost(&totalVorticity, 3*sizeof(double)));
    checkGPUError(cudaMallocHost(&linearImpulse,  3*sizeof(double)));
    checkGPUError(cudaMallocHost(&angularImpulse, 3*sizeof(double)));
    checkGPUError(cudaMallocHost(&enstrophy,      sizeof(double)));
    checkGPUError(cudaMallocHost(&kineticenergy,  sizeof(double)));
    checkGPUError(cudaMallocHost(&helicity,       sizeof(double)));
    checkGPUError(cudaMallocHost(&enstrophyF,     sizeof(double)));
    checkGPUError(cudaMallocHost(&kineticenergyF, sizeof(double)));
*/
    //Transfer particles to GPU
    S->getParticles(reinterpret_cast<double *>(cpuBuffer));
    checkGPUError(cudaMemcpy(gpuSource,cpuBuffer,mem_size,cudaMemcpyHostToDevice));

    size_t threadBlocks = (numberOfParticles + threadBlockSize - 1) / threadBlockSize;

    //Because openmp does not work in cuda files currently, we switch measurement system
    auto tStart = std::chrono::high_resolution_clock::now();
    std::cout <<"\tStep in integrate " << 1 << "\n";
    eulerKernel<threadBlockSize, unrollFactor><<<threadBlocks, threadBlockSize,0,integrateStream >>>(gpuSource,gpuTarget,numberOfParticles,S->getNu(), _dt);
    if(diagnose) {
        printf("inside pawan::gpu_euler diagnose");
        S->writenu(fdiag);
        fwrite(&t,sizeof(double),1,fdiag);
        checkGPUError(cudaMemset(totalDiag,0,17*sizeof(double)));
/*        checkGPUError(cudaMemset(totalVorticity,0,3*sizeof(double)));
        checkGPUError(cudaMemset(linearImpulse,0,3*sizeof(double)));
        checkGPUError(cudaMemset(angularImpulse,0,3*sizeof(double)));
        checkGPUError(cudaMemset(enstrophy,0,sizeof(double)));
        checkGPUError(cudaMemset(kineticenergy,0,sizeof(double)));
        checkGPUError(cudaMemset(helicity,0,sizeof(double)));
        checkGPUError(cudaMemset(enstrophyF,0,sizeof(double)));
        checkGPUError(cudaMemset(kineticenergyF,0,sizeof(double)));
        printf("totalVor = %10.5e, %10.5e, %10.5e \n",totalVorticity[0],totalVorticity[1],totalVorticity[2]);
        printf("totalLI = %10.5e, %10.5e, %10.5e \n",linearImpulse[0],linearImpulse[1],linearImpulse[2]);
        printf("totalAI = %10.5e, %10.5e, %10.5e \n",angularImpulse[0],angularImpulse[1],angularImpulse[2]);
        printf("totalE = %10.5e, totalKE = %10.5e, totalH = %10.5e, totalEf = %10.5e, totalKEf = %10.5e \n",enstrophy[0], kineticenergy[0], helicity[0], enstrophyF[0], kineticenergyF[0]);
*/
        printf("totalVor = %10.5e, %10.5e, %10.5e \n",totalDiag[0],totalDiag[1],totalDiag[2]);
        printf("totalLI = %10.5e, %10.5e, %10.5e \n",totalDiag[3],totalDiag[4],totalDiag[5]);
        printf("totalAI = %10.5e, %10.5e, %10.5e \n",totalDiag[6],totalDiag[7],totalDiag[8]);
        printf("totalE = %10.5e, totalKE = %10.5e, totalH = %10.5e, totalEf = %10.5e, totalKEf = %10.5e \n",totalDiag[9], totalDiag[10], totalDiag[11], totalDiag[12], totalDiag[13]);
        printf("Zc = %10.5e \n", totalDiag[16]);
        totalLinDiag<threadBlockSize, unrollFactor><<<threadBlocks, threadBlockSize,2*threadBlockSize*sizeof(double3),integrateStream >>>(gpuSource,numberOfParticles,totalDiag, 0);
        totalLinDiag<threadBlockSize, unrollFactor><<<threadBlocks, threadBlockSize,2*threadBlockSize*sizeof(double3),integrateStream >>>(gpuSource,numberOfParticles,totalDiag+3, 1);
        totalLinDiag<threadBlockSize, unrollFactor><<<threadBlocks, threadBlockSize,2*threadBlockSize*sizeof(double3),integrateStream >>>(gpuSource,numberOfParticles,totalDiag+6, 2);
        totalQuadDiag<threadBlockSize, unrollFactor><<<threadBlocks, threadBlockSize, 0, integrateStream >>>(gpuSource, numberOfParticles, totalDiag+9,     0);
        totalQuadDiag<threadBlockSize, unrollFactor><<<threadBlocks, threadBlockSize, 0, integrateStream >>>(gpuSource, numberOfParticles, totalDiag+10, 1);
        totalQuadDiag<threadBlockSize, unrollFactor><<<threadBlocks, threadBlockSize, 0, integrateStream >>>(gpuSource, numberOfParticles, totalDiag+11,      2);
        totalQuadDiag<threadBlockSize, unrollFactor><<<threadBlocks, threadBlockSize, 0, integrateStream >>>(gpuSource, numberOfParticles, totalDiag+12,    3);
        totalQuadDiag<threadBlockSize, unrollFactor><<<threadBlocks, threadBlockSize, 0, integrateStream >>>(gpuSource, numberOfParticles, totalDiag+13,4);
        totalLinDiag<threadBlockSize, unrollFactor><<<threadBlocks, threadBlockSize,2*threadBlockSize*sizeof(double3),integrateStream >>>(gpuSource,numberOfParticles,totalDiag+14, 3);
        cudaDeviceSynchronize();
        S->setDiagnostics(totalDiag);
        S->writediagnosis(fdiag);
/*        printf("totalVor = %10.5e, %10.5e, %10.5e \n",totalVorticity[0],totalVorticity[1],totalVorticity[2]);
        printf("totalLI = %10.5e, %10.5e, %10.5e \n",linearImpulse[0],linearImpulse[1],linearImpulse[2]);
        printf("totalAI = %10.5e, %10.5e, %10.5e \n",angularImpulse[0],angularImpulse[1],angularImpulse[2]);
        printf("totalE = %10.5e, totalKE = %10.5e, totalH = %10.5e, totalEf = %10.5e, totalKEf = %10.5e \n",enstrophy[0], kineticenergy[0], helicity[0], enstrophyF[0], kineticenergyF[0]);
*/
        printf("totalVor = %10.5e, %10.5e, %10.5e \n",totalDiag[0],totalDiag[1],totalDiag[2]);
        printf("totalLI = %10.5e, %10.5e, %10.5e \n",totalDiag[3],totalDiag[4],totalDiag[5]);
        printf("totalAI = %10.5e, %10.5e, %10.5e \n",totalDiag[6],totalDiag[7],totalDiag[8]);
        printf("totalE = %10.5e, totalKE = %10.5e, totalH = %10.5e, totalEf = %10.5e, totalKEf = %10.5e \n",totalDiag[9], totalDiag[10], totalDiag[11], totalDiag[12], totalDiag[13]);
        if(totalDiag[15]!=0)
            printf("Zc = %10.5e \n", totalDiag[14]/totalDiag[15]);
    }

    for(size_t i = 1; i<=_n; ++i){
        //switch source and target
        double4 * temp = gpuSource;
        gpuSource = gpuTarget;
        gpuTarget = temp;

        //Wait for step i-1 to finish calculating
        checkGPUError(cudaStreamSynchronize(integrateStream));

        //if not in the last step, start the next one
        if(i < _n) {
            OUT("\tStep", i+1);
            eulerKernel<threadBlockSize, unrollFactor><<<threadBlocks, threadBlockSize, 0, integrateStream >>>(gpuSource, gpuTarget, numberOfParticles,S->getNu(), _dt);
        }

        //Start copy the result of the previous calculation
        checkGPUError(cudaMemcpyAsync(cpuBuffer,gpuSource,mem_size,cudaMemcpyDeviceToHost, memoryStream));
        checkGPUError(cudaStreamSynchronize(memoryStream));

        //wait for memory copy to finish, then do all the things that need to be done on the cpu
        S->setParticles(reinterpret_cast<double *>(cpuBuffer));
        t = i*_dt; //The data is the one of the last step
        fwrite(&t,sizeof(double),1,f);
        S->write(f);  //write particles info after interaction of the last time step
        if(diagnose){
            checkGPUError(cudaMemset(totalDiag,0,17*sizeof(double)));
            printf("totalVor = %10.5e, %10.5e, %10.5e \n",totalDiag[0],totalDiag[1],totalDiag[2]);
            printf("totalLI = %10.5e, %10.5e, %10.5e \n",totalDiag[3],totalDiag[4],totalDiag[5]);
            printf("totalAI = %10.5e, %10.5e, %10.5e \n",totalDiag[6],totalDiag[7],totalDiag[8]);
            printf("totalE = %10.5e, totalKE = %10.5e, totalH = %10.5e, totalEf = %10.5e, totalKEf = %10.5e \n",totalDiag[9], totalDiag[10], totalDiag[11], totalDiag[12], totalDiag[13]);
            printf("Zc = %10.5e \n", totalDiag[16]);
            totalLinDiag<threadBlockSize, unrollFactor><<<threadBlocks, threadBlockSize,2*threadBlockSize*sizeof(double3),integrateStream >>>(gpuSource,numberOfParticles,totalDiag, 0);
            totalLinDiag<threadBlockSize, unrollFactor><<<threadBlocks, threadBlockSize,2*threadBlockSize*sizeof(double3),integrateStream >>>(gpuSource,numberOfParticles,totalDiag+3, 1);
            totalLinDiag<threadBlockSize, unrollFactor><<<threadBlocks, threadBlockSize,2*threadBlockSize*sizeof(double3),integrateStream >>>(gpuSource,numberOfParticles,totalDiag+6, 2);
            totalQuadDiag<threadBlockSize, unrollFactor><<<threadBlocks, threadBlockSize, 0, integrateStream >>>(gpuSource, numberOfParticles, totalDiag+9,     0);
            totalQuadDiag<threadBlockSize, unrollFactor><<<threadBlocks, threadBlockSize, 0, integrateStream >>>(gpuSource, numberOfParticles, totalDiag+10, 1);
            totalQuadDiag<threadBlockSize, unrollFactor><<<threadBlocks, threadBlockSize, 0, integrateStream >>>(gpuSource, numberOfParticles, totalDiag+11,      2);
            totalQuadDiag<threadBlockSize, unrollFactor><<<threadBlocks, threadBlockSize, 0, integrateStream >>>(gpuSource, numberOfParticles, totalDiag+12,    3);
            totalQuadDiag<threadBlockSize, unrollFactor><<<threadBlocks, threadBlockSize, 0, integrateStream >>>(gpuSource, numberOfParticles, totalDiag+13,4);
            totalLinDiag<threadBlockSize, unrollFactor><<<threadBlocks, threadBlockSize,2*threadBlockSize*sizeof(double3),integrateStream >>>(gpuSource,numberOfParticles,totalDiag+14, 3);
            cudaDeviceSynchronize();
            //S->setDiagnostics(totalVorticity, linearImpulse, angularImpulse, enstrophy, kineticenergy, helicity, enstrophyF, kineticenergyF);
            S->setDiagnostics(totalDiag);
            fwrite(&t,sizeof(double),1,fdiag);
            S->writediagnosis(fdiag);
/*        printf("totalVor = %10.5e, %10.5e, %10.5e \n",totalVorticity[0],totalVorticity[1],totalVorticity[2]);
        printf("totalLI = %10.5e, %10.5e, %10.5e \n",linearImpulse[0],linearImpulse[1],linearImpulse[2]);
        printf("totalAI = %10.5e, %10.5e, %10.5e \n",angularImpulse[0],angularImpulse[1],angularImpulse[2]);
        printf("totalE = %10.5e, totalKE = %10.5e, totalH = %10.5e, totalEf = %10.5e, totalKEf = %10.5e \n",enstrophy[0], kineticenergy[0], helicity[0], enstrophyF[0], kineticenergyF[0]);
*/
            printf("totalVor = %10.5e, %10.5e, %10.5e \n",totalDiag[0],totalDiag[1],totalDiag[2]);
            printf("totalLI = %10.5e, %10.5e, %10.5e \n",totalDiag[3],totalDiag[4],totalDiag[5]);
            printf("totalAI = %10.5e, %10.5e, %10.5e \n",totalDiag[6],totalDiag[7],totalDiag[8]);
            printf("totalE = %10.5e, totalKE = %10.5e, totalH = %10.5e, totalEf = %10.5e, totalKEf = %10.5e \n",totalDiag[9], totalDiag[10], totalDiag[11], totalDiag[12], totalDiag[13]);
            if(totalDiag[15]!=0)
                printf("Zc = %10.5e \n", totalDiag[14]/totalDiag[15]);
        }
    }
    fclose(f);
    auto tEnd = std::chrono::high_resolution_clock::now();

    OUT("Total Time (s)",std::chrono::duration<double>(tEnd - tStart).count());

    checkGPUError(cudaFree(gpuSource));
    checkGPUError(cudaFree(gpuTarget));
    checkGPUError(cudaFreeHost(cpuBuffer));
    checkGPUError(cudaFreeHost(totalDiag));
/*    checkGPUError(cudaFreeHost(totalVorticity));
    checkGPUError(cudaFreeHost(linearImpulse));
    checkGPUError(cudaFreeHost(angularImpulse));
    checkGPUError(cudaFreeHost(enstrophy    ));
    checkGPUError(cudaFreeHost(kineticenergy));
    checkGPUError(cudaFreeHost(helicity     ));
    checkGPUError(cudaFreeHost(enstrophyF   ));
    checkGPUError(cudaFreeHost(kineticenergyF));
*/
}


