#pragma once

#include "interaction/gpu_common.cuh"
#include "integration.h"
#include "interaction/gpu.cuh"
#include "wake/wake.h"
#include "interaction/interaction_utils_gpu.cuh"
#define XDIM 2
#define YDIM 3
#define ZDIM 4
#if !defined NULL
#define NULL 0
#endif
#define XVARNUM 1
#define YVARNUM 2
#define ZVARNUM 3


namespace pawan{
    template<int threadBlockSize = 128, int unrollFactor = 1>
    class gpu_int : public __integration{

    private:
        void writeszl(void *fileHandle, double *p, double &t, int &numberOfParticles);

    public:
        gpu_int(const double &t, const size_t &n);
        gpu_int();

        ~gpu_int() = default;

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
pawan::gpu_int<threadBlockSize,unrollFactor>::gpu_int(const double &t, const size_t &n):__integration(t, n){}
template<int threadBlockSize, int unrollFactor>
pawan::gpu_int<threadBlockSize,unrollFactor>::gpu_int():__integration(){}

template<int threadBlockSize = 128, int unrollFactor = 1>
__global__ void stepKernel(const double4 *source, double4 *target, double3 *rates, const size_t N, const double nu,const double dt) {

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
        rates[2 * index] = ownVelocity;
        rates[2 * index + 1] = ownRetVorticity;
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
__global__ void rk4stepKernel(const double4 *source, const double4 *x, double4 *target, double3 *rates, const size_t N, const double nu,const double dt) {

    double4 xPosition, xVorticity;
    double4 ownPosition, ownVorticity;
    double3 xVelocity = {0,0,0}, xRetVorticity = {0,0,0};

    size_t index = blockIdx.x * threadBlockSize + threadIdx.x;

    //cache own particle if index in bounds
    if(index < N){
        xPosition = x[2 * index];
        xVorticity = x[2 * index + 1];
        ownPosition = source[2 * index];
        ownVorticity = source[2 * index + 1];
    }

    interact_with_all<threadBlockSize,unrollFactor>(x, N, nu, xPosition, xVorticity, index, xVelocity, xRetVorticity);

    if(index < N) {
        rates[2 * index] = xVelocity;
        rates[2 * index + 1] = xRetVorticity;
        //do the integration step and write result to target
        scale(dt, xVelocity);
        add(ownPosition,xVelocity);
        scale(dt,xRetVorticity);
        add(ownVorticity, xRetVorticity);

        target[2 * index] = ownPosition;
        target[2 * index + 1] = ownVorticity;
    }

}

template<int threadBlockSize = 128, int unrollFactor = 1>
__global__ void rk4finalstepKernel(const double4 *source, double4 *target,
                                   const double3 *k1, const double3 *k2, const double3 *k3, const double3 *k4,
                                   const size_t N, const double dt) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < N) {
        target[2 * tid].x     = source[2 * tid].x     + (dt/6.0) * (k1[2 * tid].x     + 2.0*k2[2 * tid].x     + 2.0*k3[2 * tid].x     + k4[2 * tid].x);
        target[2 * tid].y     = source[2 * tid].y     + (dt/6.0) * (k1[2 * tid].y     + 2.0*k2[2 * tid].y     + 2.0*k3[2 * tid].y     + k4[2 * tid].y);
        target[2 * tid].z     = source[2 * tid].z     + (dt/6.0) * (k1[2 * tid].z     + 2.0*k2[2 * tid].z     + 2.0*k3[2 * tid].z     + k4[2 * tid].z);
        target[2 * tid].w     = source[2 * tid].w;
        target[2 * tid + 1].x = source[2 * tid + 1].x + (dt/6.0) * (k1[2 * tid + 1].x + 2.0*k2[2 * tid + 1].x + 2.0*k3[2 * tid + 1].x + k4[2 * tid + 1].x);
        target[2 * tid + 1].y = source[2 * tid + 1].y + (dt/6.0) * (k1[2 * tid + 1].y + 2.0*k2[2 * tid + 1].y + 2.0*k3[2 * tid + 1].y + k4[2 * tid + 1].y);
        target[2 * tid + 1].z = source[2 * tid + 1].z + (dt/6.0) * (k1[2 * tid + 1].z + 2.0*k2[2 * tid + 1].z + 2.0*k3[2 * tid + 1].z + k4[2 * tid + 1].z);
        target[2 * tid + 1].w = source[2 * tid + 1].w;
    }
}

template<int threadBlockSize = 128, int unrollFactor = 1>
void rk4Step(const double4 *source, double4 *target, const size_t N, const double nu, const double dt,
             double4* x1, double4* x2, double4* x3,
             double3* k1, double3* k2, double3* k3, double3* k4,
             cudaStream_t stream, const size_t threadBlocks) {

    // k1 = f(x,t)
    // x1 = x + 0.5*dt*k1
    stepKernel<threadBlockSize,unrollFactor><<<threadBlocks, threadBlockSize,0,stream>>>(source,x1,k1,N, nu, 0.5*dt);

    // k2 = f(x1, t+0.5*dt)
    // x2 = x + 0.5*dt*k2
    rk4stepKernel<threadBlockSize,unrollFactor><<<threadBlocks, threadBlockSize,0,stream>>>(source,x1,x2,k2,N, nu, 0.5*dt);

    // k3 = f(x2, t+0.5*dt)
    // x3 = x + dt*k3
    rk4stepKernel<threadBlockSize,unrollFactor><<<threadBlocks, threadBlockSize,0,stream>>>(source,x2,x3,k3,N, nu, dt);

    // k4 = f(x3, t+dt)
    //x2 used as dummy input since it is no longer required
    rk4stepKernel<threadBlockSize,unrollFactor><<<threadBlocks, threadBlockSize,0,stream>>>(source,x3,x2,k4,N, nu, dt);

    rk4finalstepKernel<threadBlockSize,unrollFactor><<<threadBlocks, threadBlockSize, 0, stream>>>(source, target, k1, k2, k3, k4, N, dt);
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

template<int threadBlockSize = 128, int unrollFactor = 1>
void runDiag(const size_t threadBlocks, const double4 *gpuSource, const size_t numberOfParticles, double *totalDiag, cudaStream_t stream) {
    totalLinDiag<threadBlockSize, unrollFactor><<<threadBlocks, threadBlockSize,2*threadBlockSize*sizeof(double3),stream >>>(gpuSource,numberOfParticles,totalDiag, 0);
    totalLinDiag<threadBlockSize, unrollFactor><<<threadBlocks, threadBlockSize,2*threadBlockSize*sizeof(double3),stream >>>(gpuSource,numberOfParticles,totalDiag+3, 1);
    totalLinDiag<threadBlockSize, unrollFactor><<<threadBlocks, threadBlockSize,2*threadBlockSize*sizeof(double3),stream >>>(gpuSource,numberOfParticles,totalDiag+6, 2);
    totalQuadDiag<threadBlockSize, unrollFactor><<<threadBlocks, threadBlockSize, 0, stream >>>(gpuSource, numberOfParticles, totalDiag+9,     0);
    totalQuadDiag<threadBlockSize, unrollFactor><<<threadBlocks, threadBlockSize, 0, stream >>>(gpuSource, numberOfParticles, totalDiag+10, 1);
    totalQuadDiag<threadBlockSize, unrollFactor><<<threadBlocks, threadBlockSize, 0, stream >>>(gpuSource, numberOfParticles, totalDiag+11,      2);
    totalQuadDiag<threadBlockSize, unrollFactor><<<threadBlocks, threadBlockSize, 0, stream >>>(gpuSource, numberOfParticles, totalDiag+12,    3);
    totalQuadDiag<threadBlockSize, unrollFactor><<<threadBlocks, threadBlockSize, 0, stream >>>(gpuSource, numberOfParticles, totalDiag+13,4);
    totalLinDiag<threadBlockSize, unrollFactor><<<threadBlocks, threadBlockSize,2*threadBlockSize*sizeof(double3),stream >>>(gpuSource,numberOfParticles,totalDiag+14, 3);
    totalLinDiag<threadBlockSize, unrollFactor><<<threadBlocks, threadBlockSize,2*threadBlockSize*sizeof(double3),stream >>>(gpuSource,numberOfParticles,totalDiag+17, 4);
}

template<int threadBlockSize, int unrollFactor>
void pawan::gpu_int<threadBlockSize,unrollFactor>::writeszl(void *fileHandle, double *p, double &t, int &numberOfParticles){
    int32_t zoneHandle;
    int varTypes[9] = {FieldDataType_Double,FieldDataType_Double,FieldDataType_Double,FieldDataType_Double,
                       FieldDataType_Double,FieldDataType_Double,FieldDataType_Double,FieldDataType_Double,
                       FieldDataType_Double};
    std::string s;
    std::stringstream convert;
    convert << t;
    s = convert.str();
    std::vector<int32_t> valueLocations(9, 1);
    int32_t tmp;
    tmp = tecZoneCreateIJK(fileHandle, s.c_str(), numberOfParticles, 1, 1, &varTypes[0], 0,&valueLocations[0], 0, 0, 0, 0, &zoneHandle);
    tmp = tecZoneVarWriteDoubleValues(fileHandle, zoneHandle, 1, 0, numberOfParticles, &p[0]);
    tmp = tecZoneVarWriteDoubleValues(fileHandle, zoneHandle, 2, 0, numberOfParticles, &p[numberOfParticles]);
    tmp = tecZoneVarWriteDoubleValues(fileHandle, zoneHandle, 3, 0, numberOfParticles, &p[2*numberOfParticles]);
    tmp = tecZoneVarWriteDoubleValues(fileHandle, zoneHandle, 4, 0, numberOfParticles, &p[4*numberOfParticles]);
    tmp = tecZoneVarWriteDoubleValues(fileHandle, zoneHandle, 5, 0, numberOfParticles, &p[5*numberOfParticles]);
    tmp = tecZoneVarWriteDoubleValues(fileHandle, zoneHandle, 6, 0, numberOfParticles, &p[6*numberOfParticles]);
    tmp = tecZoneVarWriteDoubleValues(fileHandle, zoneHandle, 7, 0, numberOfParticles, &p[3*numberOfParticles]); //radius
    tmp = tecZoneVarWriteDoubleValues(fileHandle, zoneHandle, 8, 0, numberOfParticles, &p[7*numberOfParticles]); //vol
    tmp = tecZoneVarWriteDoubleValues(fileHandle, zoneHandle, 9, 0, numberOfParticles, &p[8*numberOfParticles]); //Vor_strength
    tmp =   tecZoneSetUnsteadyOptions(fileHandle, zoneHandle, t, 1);
}

template<int threadBlockSize, int unrollFactor>
void pawan::gpu_int<threadBlockSize,unrollFactor>::integrate(pawan::__system *S, pawan::__io *IO,
                                                             NetworkInterfaceTCP<OPawanRecvData, OPawanSendData> *networkCommunicatorTest,
                                                             bool diagnose) {
    //Because openmp does not work in cuda files currently, we switch measurement system
    auto tStart = std::chrono::high_resolution_clock::now();
    FILE *fdiag = IO->create_binary_file(".diagnosis");
    OPawanRecvData opawanrecvdata;
    networkCommunicatorTest->getrecieveBuffer(opawanrecvdata);
    _t = opawanrecvdata.t;
    FILE *f = IO->create_binary_file(".wake");

    cudaStream_t integrateStream; cudaStreamCreate(&integrateStream);
    //Memory allocations:
    //two GPU buffers so that in each step the result can be written
    //without having to wait on all threads finishing
    //one pinned memory buffer on the cpu for copying states back
    int numberOfParticles;
    int maxnumberOfParticles = S->totalmaxParticles();
    size_t mem_size = maxnumberOfParticles * 2 * sizeof(double4);
    size_t mem_sized3 = maxnumberOfParticles * 2 * sizeof(double3);
    double4 *gpuSource, *gpuTarget, *cpuBuffer;
    double3 *divfreevor;
    checkGPUError(cudaMallocHost(&cpuBuffer, mem_size));
    checkGPUError(cudaMalloc(&gpuSource, mem_size));
    checkGPUError(cudaMalloc(&gpuTarget, mem_size));
    checkGPUError(cudaMalloc(&divfreevor, mem_sized3));
    double4 *x1, *x2, *x3;
    double3 *k1, *k2, *k3, *k4;
    checkGPUError(cudaMalloc(&x1, mem_size));checkGPUError(cudaMalloc(&x2, mem_size));checkGPUError(cudaMalloc(&x3, mem_size));
    checkGPUError(cudaMalloc(&k1, mem_sized3));checkGPUError(cudaMalloc(&k2, mem_sized3));checkGPUError(cudaMalloc(&k3, mem_sized3));checkGPUError(cudaMalloc(&k4, mem_sized3));
    double *totalDiag;
    checkGPUError(cudaMallocHost(&totalDiag, 20*sizeof(double)));

    size_t threadBlocks;// = (numberOfParticles + threadBlockSize - 1) / threadBlockSize;

    std::string szlfilename = IO->getSzlFile();
    std::string variables("x y z vor_x vor_y vor_z radius vol Vor_strength");//9 variables
    int varTypes[9] = {FieldDataType_Double,FieldDataType_Double,FieldDataType_Double,FieldDataType_Double,
                       FieldDataType_Double,FieldDataType_Double,FieldDataType_Double,FieldDataType_Double,
                       FieldDataType_Double};
    void* fileHandle;
    int32_t res = tecFileWriterOpen(szlfilename.c_str(),"IJK Ordered Zone",variables.c_str(),1,0,FieldDataType_Double,0,&fileHandle);
    double *p = (double*) malloc(maxnumberOfParticles * 9 * sizeof(double));

    size_t stepnum = 0;
    double relax_factor = 0.5;   //f*delta_t from Pedrizetti's equation
    while(_t <= opawanrecvdata.tfinal){
        OUT("\tTime",_t);
        OUT("\tStepNum",stepnum);

        S->getParticles(reinterpret_cast<double *>(cpuBuffer));
        checkGPUError(cudaMemcpy(gpuSource,cpuBuffer,mem_size,cudaMemcpyHostToDevice));
        numberOfParticles = S->amountParticles();
        threadBlocks = (numberOfParticles + threadBlockSize - 1) / threadBlockSize;
        printf("\tnumberOfParticles = %10d",numberOfParticles);

        //stepKernel<threadBlockSize, unrollFactor><<<threadBlocks, threadBlockSize, 0, integrateStream >>>(gpuSource, gpuTarget, rates,numberOfParticles,S->getNu(), _dt);
        rk4Step<threadBlockSize, unrollFactor>(gpuSource,gpuTarget,numberOfParticles,S->getNu(), opawanrecvdata.deltat, x1, x2, x3, k1, k2, k3, k4, integrateStream, threadBlocks);
        checkGPUError(cudaMemcpy(cpuBuffer,gpuTarget,mem_size,cudaMemcpyDeviceToHost));
        S->setParticles(reinterpret_cast<double *>(cpuBuffer));

        int transient_steps = opawanrecvdata.transientsteps;
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
        S->getParticles_arr(p);
        writeszl(fileHandle,p,_t,numberOfParticles);

        OPawanSendData opawansenddata;//create it once outside the loop and should be good
        S->getInflow(&opawanrecvdata,&opawansenddata);
        networkCommunicatorTest->send_data(opawansenddata);
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
    free(p);
    res = tecFileWriterClose(&fileHandle);
    auto tEnd = std::chrono::high_resolution_clock::now();

    OUT("Total Time (s)",std::chrono::duration<double>(tEnd - tStart).count());

    checkGPUError(cudaFree(gpuSource));
    checkGPUError(cudaFree(gpuTarget));
    checkGPUError(cudaFreeHost(cpuBuffer));
}
template<int threadBlockSize, int unrollFactor>
void pawan::gpu_int<threadBlockSize,unrollFactor>::integrate(pawan::__system *S, pawan::__io *IO, bool diagnose) {
    FILE *f = IO->create_binary_file(".wake");
    FILE *fdiag = IO->create_binary_file(".diagnosis");
    double t = 0.0;
    fwrite(&t,sizeof(double),1,f);
    S->write(f);  //write particles info as is

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
    size_t mem_sized3 = maxnumberOfParticles * 2 * sizeof(double3);
    double4 *gpuSource, *gpuTarget, *cpuBuffer;
    double3 *rates;
    checkGPUError(cudaMallocHost(&cpuBuffer, mem_size));
    checkGPUError(cudaMalloc(&gpuSource, mem_size));
    checkGPUError(cudaMalloc(&gpuTarget, mem_size));
    checkGPUError(cudaMalloc(&rates, mem_sized3));
    double4 *x1, *x2, *x3;
    double3 *k1, *k2, *k3, *k4;
    checkGPUError(cudaMalloc(&x1, mem_size));checkGPUError(cudaMalloc(&x2, mem_size));checkGPUError(cudaMalloc(&x3, mem_size));
    checkGPUError(cudaMalloc(&k1, mem_sized3));checkGPUError(cudaMalloc(&k2, mem_sized3));checkGPUError(cudaMalloc(&k3, mem_sized3));checkGPUError(cudaMalloc(&k4, mem_sized3));
    double *totalDiag;
    checkGPUError(cudaMallocHost(&totalDiag, 20*sizeof(double)));

    //Transfer particles to GPU
    S->getParticles(reinterpret_cast<double *>(cpuBuffer));
    checkGPUError(cudaMemcpy(gpuSource,cpuBuffer,mem_size,cudaMemcpyHostToDevice));

    size_t threadBlocks = (numberOfParticles + threadBlockSize - 1) / threadBlockSize;

    if(diagnose) {
        S->writenu(fdiag);
        fwrite(&t,sizeof(double),1,fdiag);
        checkGPUError(cudaMemset(totalDiag,0.0,20*sizeof(double)));
        runDiag(threadBlocks, gpuSource, numberOfParticles,totalDiag, integrateStream);
        cudaDeviceSynchronize();
        S->printdiagnostics(totalDiag);
        S->setDiagnostics(totalDiag);
        S->writediagnosis(fdiag);
    }

    //Because openmp does not work in cuda files currently, we switch measurement system
    auto tStart = std::chrono::high_resolution_clock::now();
    for(size_t i = 1; i<=_n; ++i){

        //Wait for step i-1 to finish calculating
        checkGPUError(cudaStreamSynchronize(integrateStream));

        //if not in the last step, start the next one
        if(i < _n) {
            OUT("\tStep", i);
            //stepKernel<threadBlockSize, unrollFactor><<<threadBlocks, threadBlockSize, 0, integrateStream >>>(gpuSource, gpuTarget, rates,numberOfParticles,S->getNu(), _dt);
            rk4Step<threadBlockSize, unrollFactor>(gpuSource,gpuTarget,numberOfParticles,S->getNu(), _dt, x1, x2, x3, k1, k2, k3, k4, integrateStream, threadBlocks);
            //cudaDeviceSynchronize();
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
            checkGPUError(cudaMemset(totalDiag,0.0,20*sizeof(double)));
            cudaDeviceSynchronize();
            runDiag(threadBlocks, gpuTarget, numberOfParticles,totalDiag, integrateStream);
            cudaDeviceSynchronize();
            S->printdiagnostics(totalDiag);
            S->setDiagnostics(totalDiag);
            fwrite(&t,sizeof(double),1,fdiag);
            S->writediagnosis(fdiag);
        }
        //switch source and target
        double4 * temp = gpuSource;
        gpuSource = gpuTarget;
        gpuTarget = temp;
    }
    fclose(f);
    auto tEnd = std::chrono::high_resolution_clock::now();

    OUT("Total Time (s)",std::chrono::duration<double>(tEnd - tStart).count());

    checkGPUError(cudaFree(gpuSource));
    checkGPUError(cudaFree(gpuTarget));
    checkGPUError(cudaFreeHost(cpuBuffer));
    checkGPUError(cudaFreeHost(totalDiag));
    checkGPUError(cudaFree(x1));checkGPUError(cudaFree(x2));checkGPUError(cudaFree(x3));
    checkGPUError(cudaFree(k1));checkGPUError(cudaFree(k2));checkGPUError(cudaFree(k3));checkGPUError(cudaFree(k4));
}


