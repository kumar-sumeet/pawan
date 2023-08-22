
#include "gpu_euler.cuh"
#include "interaction/gpu.cuh"
#include "interaction/gpu_common.cuh"

void resizeToFit(double4 *cpu, double4 *gpu1, double4 *gpu2, size_t &size, int particles);

pawan::gpu_euler::gpu_euler(const double &t, const size_t &n):__integration(t,n){}


__global__ void eulerKernel(const double4 *source, double4 *target, const size_t N, const double nu, double dt) {

    double4 ownPosition, ownVorticity;
    double3 ownVelocity = {0,0,0}, ownRetVorticity = {0,0,0};

    size_t index = blockIdx.x * threadBlockSize + threadIdx.x;

    //cache own particle if index in bounds
    if(index < N){
        ownPosition = source[2 * index];
        ownVorticity = source[2 * index + 1];
    }

    interact_with_all(source, N, nu, ownPosition, ownVorticity, index, ownVelocity, ownRetVorticity);

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


void pawan::gpu_euler::integrate(pawan::__system *S, pawan::__io *IO,
                                 NetworkInterfaceTCP<OPawanRecvData, OPawanSendData> *networkCommunicatorTest,
                                 bool diagnose) {
    double tStart = TIME();
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
    size_t mem_size = numberOfParticles * 2 * sizeof(double4);

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
        mem_size = numberOfParticles * 2 * sizeof(double4);

        resizeToFit(cpuBuffer, gpuSource, gpuTarget, mem_size, S->amountParticles());
        S->getParticles(reinterpret_cast<double *>(cpuBuffer));
        checkGPUError(cudaMemcpy(gpuSource,cpuBuffer,mem_size,cudaMemcpyHostToDevice));

        size_t threadBlocks = (numberOfParticles + threadBlockSize - 1) / threadBlockSize;
        eulerKernel<<<threadBlocks, threadBlockSize>>>(gpuSource,gpuTarget,numberOfParticles,S->getNu(), _dt);

        checkGPUError(cudaMemcpy(cpuBuffer,gpuTarget,mem_size,cudaMemcpyDeviceToHost));
        S->setParticles(reinterpret_cast<double *>(cpuBuffer));

        //S->relax();
        if(diagnose){
            S->diagnose();
            fwrite(&_t,sizeof(double),1,fdiag);
            S->writediagnosis(fdiag);
        }
        S->updateVinfEffect(opawanrecvdata.Vinf,opawanrecvdata.deltat);
        //S->updateBoundVorEffect(&opawanrecvdata,_dt);
        fwrite(&_t,sizeof(double),1,f);
        S->write(f);  //write particles info after interaction in this time step

        OPawanSendData opawansenddata;//create it once outside the loop and should be good
        S->getInflow(&opawanrecvdata,&opawansenddata);
        networkCommunicatorTest->send_data(opawansenddata);

        //S->diagnose();
        if(_t <= (opawanrecvdata.tfinal - 1*opawanrecvdata.deltat)){ //run till end of dymore sim
            networkCommunicatorTest->recieve_data(opawanrecvdata);
            S->addParticles(&opawanrecvdata);
            _t = opawanrecvdata.t;
        }
        else
            break;
        stepnum = stepnum+1;
    }
    fclose(f);
    double tEnd = TIME();
    OUT("Total Time (s)",tEnd - tStart);

    checkGPUError(cudaFree(gpuSource));
    checkGPUError(cudaFree(gpuTarget));
    checkGPUError(cudaFreeHost(cpuBuffer));
}

void pawan::gpu_euler::integrate(pawan::__system *S, pawan::__io *IO, bool diagnose) {
    FILE *f = IO->create_binary_file(".wake");
    FILE *fdiag = IO->create_binary_file(".diagnosis");
    double t = 0.0;
    fwrite(&t,sizeof(double),1,f);
    S->write(f);  //write particles info as is
    if(diagnose) {
        S->writenu(fdiag);
        fwrite(&t,sizeof(double),1,fdiag);
        S->diagnose();
        S->writediagnosis(fdiag);
    }

    //Create two cuda streams so that integration and memory copies can happen at the same time
    cudaStream_t memoryStream, integrateStream;
    cudaStreamCreate(&memoryStream);
    cudaStreamCreate(&integrateStream);

    //Memory allocations:
    //two GPU buffers so that in each step the result can be written
    //without having to wait on all threads finishing
    //one pinned memory buffer on the cpu for copying states back
    //TODO allow size to change? only in network version?
    int numberOfParticles = S->amountParticles();
    size_t mem_size = numberOfParticles * 2 * sizeof(double4);

    double4 *gpuSource, *gpuTarget, *cpuBuffer;
    checkGPUError(cudaMallocHost(&cpuBuffer, mem_size));
    checkGPUError(cudaMalloc(&gpuSource, mem_size));
    checkGPUError(cudaMalloc(&gpuTarget, mem_size));

    //Transfer particles to GPU
    S->getParticles(reinterpret_cast<double *>(cpuBuffer));
    checkGPUError(cudaMemcpy(gpuSource,cpuBuffer,mem_size,cudaMemcpyHostToDevice));

    size_t threadBlocks = (numberOfParticles + threadBlockSize - 1) / threadBlockSize;

    double tStart = TIME();
    std::cout <<"\tStep " << 1 << "\n";
    eulerKernel<<<threadBlocks, threadBlockSize,0,integrateStream >>>(gpuSource,gpuTarget,numberOfParticles,S->getNu(), _dt);

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
            eulerKernel<<<threadBlocks, threadBlockSize, 0, integrateStream >>>(gpuSource, gpuTarget, numberOfParticles,
                                                                                S->getNu(), _dt);
        }

        //Start copy the result of the previous calculation
        checkGPUError(cudaMemcpyAsync(cpuBuffer,gpuSource,mem_size,cudaMemcpyDeviceToHost, memoryStream));

        //wait for memory copy to finish, then do all the things that need to be done on the cpu
        checkGPUError(cudaStreamSynchronize(memoryStream));
        S->setParticles(reinterpret_cast<double *>(cpuBuffer));
        t = i*_dt; //The data is the one of the last step
        fwrite(&t,sizeof(double),1,f);
        S->write(f);  //write particles info after interaction of the last time step

        if(diagnose){
            S->diagnose();
            fwrite(&t,sizeof(double),1,fdiag);
            S->writediagnosis(fdiag);
        }
    }
    fclose(f);
    double tEnd = TIME();
    OUT("Total Time (s)",tEnd - tStart);

    checkGPUError(cudaFree(gpuSource));
    checkGPUError(cudaFree(gpuTarget));
    checkGPUError(cudaFreeHost(cpuBuffer));
}



void resizeToFit(double4 *cpu, double4 *gpu1, double4 *gpu2, size_t &size, int particles) {

    size_t neededsize = particles * 2 * sizeof(double4);

    if(neededsize > size){
        while(neededsize > size){
            size *= 1.5;
        }

        checkGPUError(cudaFree(gpu1));
        checkGPUError(cudaFree(gpu2));
        checkGPUError(cudaFreeHost(cpu));

        checkGPUError(cudaMallocHost(&cpu, size));
        checkGPUError(cudaMalloc(&gpu1, size));
        checkGPUError(cudaMalloc(&gpu2, size));

    }


}
