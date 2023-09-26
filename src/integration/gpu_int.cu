#include "gpu_int.cuh"

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