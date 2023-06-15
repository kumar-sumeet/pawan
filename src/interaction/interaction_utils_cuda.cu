
#include <cublas_v2.h>
#include <cuda.h>
#include <stdio.h>
#include "src/integration/integration.h"
#include "src/io/io.h"
#include "src/utils/timing_utils.h"
#include "src/wake/wake_struct.h"

#define BLOCKSIZE 512
#define SOFTENING_CUDA 1e-12f
#define FACTOR1 0.5
#define FACTOR2 1.0

__device__ void KERNEL_CUDA(const double& rho,
                            const double& sigma,
                            double& q,
                            double& F,
                            double& Z) {
    double rho_bar = rho / sigma;
    double sig3 = sigma * sigma * sigma;
    double phi = 0.25 * M_1_PI * erf(M_SQRT1_2 * rho_bar) / sig3;
    Z = 0.5 * exp(-0.5 * rho_bar * rho_bar) / sig3 / pow(M_PI, 1.5);
    q = (phi / rho_bar - Z) / (rho_bar * rho_bar);
    F = (Z - 3 * q) / (rho * rho);
};

__device__ void VELOCITY_CUDA(const double& kernel,
                              const double* vorticity,
                              const float3 displacement,
                              float3& velocity) {
    velocity.x = (vorticity[1] * displacement.z - vorticity[2] * displacement.y) * kernel;
    velocity.y = (vorticity[2] * displacement.x - vorticity[0] * displacement.z) * kernel;
    velocity.z = (vorticity[0] * displacement.y - vorticity[1] * displacement.x) * kernel;
};

__device__ void VORSTRETCH_CUDA(const double& q,
                                const double& F,
                                const double* source_vorticity,
                                const double* target_vorticity,
                                const float3 displacement,
                                float3& retvorcity) {
    double trgXsrc0, trgXsrc1, trgXsrc2;
    trgXsrc0 = target_vorticity[1] * source_vorticity[2] - target_vorticity[2] * source_vorticity[1];
    trgXsrc1 = target_vorticity[2] * source_vorticity[0] - target_vorticity[0] * source_vorticity[2];
    trgXsrc2 = target_vorticity[0] * source_vorticity[1] - target_vorticity[1] * source_vorticity[0];

    double roaxa = 0.0;
    roaxa += displacement.x * trgXsrc0;
    roaxa += displacement.y * trgXsrc1;
    roaxa += displacement.z * trgXsrc2;

    retvorcity.x += ((trgXsrc0 * q) + (displacement.x * F * roaxa));
    retvorcity.y += ((trgXsrc1 * q) + (displacement.y * F * roaxa));
    retvorcity.z += ((trgXsrc2 * q) + (displacement.z * F * roaxa));
};

__device__ void DIFFUSION_CUDA(const double& nu,
                               const double& sigma,
                               const double& Z,
                               const double* source_vorticity,
                               const double* target_vorticity,
                               const double& source_volume,
                               const double& target_volume,
                               float3& retvorcity) {
    double sig12 = 0.5 * sigma * sigma;
    retvorcity.x += ((source_vorticity[0] * target_volume) - (target_vorticity[0] * source_volume)) * (Z * nu / sig12);
    retvorcity.y += ((source_vorticity[1] * target_volume) - (target_vorticity[1] * source_volume)) * (Z * nu / sig12);
    retvorcity.z += ((source_vorticity[2] * target_volume) - (target_vorticity[2] * source_volume)) * (Z * nu / sig12);
}

__device__ void INTERACT_CUDA(
    const double& nu,
    const double& s_source,
    const double& s_target,
    const double* r_source,
    const double* r_target,
    const double* a_source,
    const double* a_target,
    const double& v_source,
    const double& v_target,
    double* dr_source,
    double* da_source) {
    // kenerl computation
    float3 displacement = make_float3(r_target[0] - r_source[0], r_target[1] - r_source[1], r_target[2] - r_source[2]);
    double rho = std::sqrt(displacement.x * displacement.x + displacement.y * displacement.y + displacement.z * displacement.z + SOFTENING);
    double q = 0.0, F = 0.0, Z = 0.0;
    double sigma = std::sqrt(s_source * s_source + s_target * s_target) / 2.0;

    // velocity computation
    float3 dr = make_float3(0.0, 0.0, 0.0);
    KERNEL_CUDA(rho, sigma, q, F, Z);
    VELOCITY_CUDA(q, a_source, displacement, dr);
    dr_source[0] += dr.x;
    dr_source[1] += dr.y;
    dr_source[2] += dr.z;

    // Rate of change of vorticity computation
    float3 da = make_float3(0.0, 0.0, 0.0);
    VORSTRETCH_CUDA(q, F, a_source, a_target, displacement, da);
    DIFFUSION_CUDA(nu, sigma, Z, a_source, a_target, v_source, v_target, da);
    da_source[0] -= da.x;
    da_source[1] -= da.y;
    da_source[2] -= da.z;
}

__global__ void setStates_cuda(pawan::wake_cuda w, const double* state) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < w.size / 2) {
        w.position[tid] = state[tid];
        w.vorticity[tid] = state[tid + w.size / 2];
    }
}

__global__ void getStates_cuda(pawan::wake_cuda w, double* state) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < w.size / 2) {
        state[tid] = w.position[tid];
        state[tid + w.size / 2] = w.vorticity[tid];
    }
}

__global__ void getRates_cuda(pawan::wake_cuda w, double* rate) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < w.size / 2) {
        rate[tid] = w.velocity[tid];
        rate[tid + w.size / 2] = w.retvorcity[tid];
    }
}

__global__ void clear(pawan::wake_cuda w) {
    size_t numDimensions = w.numDimensions;
    size_t numParticles = w.numParticles;
    for (size_t i = 0; i < numParticles; i++)
        for (size_t j = 0; j < numDimensions; j++)
            w.velocity[i * numDimensions + j] = w.retvorcity[i * numDimensions + j] = 0.0;
}

__global__ void interact_cuda(pawan::wake_cuda w) {
    int i_src = blockIdx.x * blockDim.x + threadIdx.x;
    if (i_src < w.numParticles) {
        size_t numDimensions = w.numDimensions;

        const double* r_src = &(w.position[i_src * numDimensions]);
        const double* a_src = &(w.vorticity[i_src * numDimensions]);
        double* dr_src = &(w.velocity[i_src * numDimensions]);
        double* da_src = &(w.retvorcity[i_src * numDimensions]);
        double s_src = w.radius[i_src];
        double v_src = w.volume[i_src];

        __shared__ double r_trgs[BLOCKSIZE * 3];
        __shared__ double a_trgs[BLOCKSIZE * 3];
        __shared__ double s_trgs[BLOCKSIZE];
        __shared__ double v_trgs[BLOCKSIZE];
        for (int block = 0; block < gridDim.x; block++) {
            int index = threadIdx.x + block * BLOCKSIZE;
            r_trgs[threadIdx.x * numDimensions] = w.position[index * numDimensions];
            r_trgs[threadIdx.x * numDimensions + 1] = w.position[index * numDimensions + 1];
            r_trgs[threadIdx.x * numDimensions + 2] = w.position[index * numDimensions + 2];

            a_trgs[threadIdx.x * numDimensions] = w.vorticity[index * numDimensions];
            a_trgs[threadIdx.x * numDimensions + 1] = w.vorticity[index * numDimensions + 1];
            a_trgs[threadIdx.x * numDimensions + 2] = w.vorticity[index * numDimensions + 2];

            s_trgs[threadIdx.x] = w.radius[index];
            v_trgs[threadIdx.x] = w.volume[index];
            __syncthreads();
#pragma unroll
            for (size_t i_trg = 0; i_trg < w.numParticles; i_trg++) {
                const double* r_trg = &(r_trgs[i_trg * numDimensions]);
                const double* a_trg = &(a_trgs[i_trg * numDimensions]);
                double s_trg = s_trgs[i_trg];
                double v_trg = v_trgs[i_trg];

                INTERACT_CUDA(w._nu, s_src, s_trg, r_src, r_trg, a_src, a_trg, v_src, v_trg, dr_src, da_src);
            }
        }
    }
}

__global__ void rk4_process(const double dt, double* x, const double* k, const double* d_states, const double factor, const int len) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < len) {
        x[tid] = k[tid];
        x[tid] *= factor * dt;
        x[tid] += d_states[tid];
    }
}

__global__ void rk4_final(const double dt, double* d_states, double* k1, double* k2, const double* k3, const double* k4, const int len) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < len) {
        k1[tid] += k4[tid];
        k1[tid] *= dt / 6.;

        k2[tid] += k3[tid];
        k2[tid] *= dt / 3.;

        k1[tid] += k2[tid];

        d_states[tid] += k1[tid];
    }
}

__global__ void scale(double* rates, const double dt, const int len) {
    for (int i = 0; i < len; i++) {
        rates[i] *= dt;
    }
}

__global__ void add(double* states, double* rates, const int len) {
    for (int i = 0; i < len; i++) {
        states[i] += rates[i];
    }
}

void step_cuda(const double dt, pawan::wake_cuda* w, double* d_states, double* rates, const int len) {
    int numBlocks_states = (len / 2 + BLOCKSIZE - 1) / BLOCKSIZE;
    int numBlocks_interact = (w->numParticles + BLOCKSIZE - 1) / BLOCKSIZE;

    clear<<<1, 1>>>(*w);
    interact_cuda<<<numBlocks_interact, BLOCKSIZE>>>(*w);
    getRates_cuda<<<numBlocks_states, BLOCKSIZE>>>(*w, rates);
    scale<<<1, 1>>>(rates, dt, len);
    add<<<1, 1>>>(d_states, rates, len);
    setStates_cuda<<<numBlocks_states, BLOCKSIZE>>>(*w, d_states);
}

extern "C" void cuda_step_wrapper(const double _dt, pawan::wake_struct* w, double* state_array) {
    // TODO: replace cudaMallocManaged with cudaMalloc for wake_cuda
    pawan::wake_cuda cuda_wake;
    cuda_wake.size = w->size;
    cuda_wake.numParticles = w->numParticles;
    cudaMallocManaged(&cuda_wake.position, sizeof(double) * w->size);
    cudaMallocManaged(&cuda_wake.velocity, sizeof(double) * w->size);
    cudaMallocManaged(&cuda_wake.vorticity, sizeof(double) * w->size);
    cudaMallocManaged(&cuda_wake.retvorcity, sizeof(double) * w->size);
    cudaMallocManaged(&cuda_wake.radius, sizeof(double) * w->numParticles);
    cudaMallocManaged(&cuda_wake.volume, sizeof(double) * w->numParticles);
    cudaMallocManaged(&cuda_wake.birthstrength, sizeof(double) * w->numParticles);

    // initialize wake on gpu
    for (size_t i = 0; i < w->numParticles; i++) {
        cuda_wake.radius[i] = w->radius[i];
        cuda_wake.volume[i] = w->volume[i];
        cuda_wake.birthstrength[i] = w->birthstrength[i];
        for (size_t j = 0; j < w->numDimensions; j++) {
            cuda_wake.position[i * w->numDimensions + j] = w->position[i][j];
            cuda_wake.velocity[i * w->numDimensions + j] = w->velocity[i][j];
            cuda_wake.vorticity[i * w->numDimensions + j] = w->vorticity[i][j];
            cuda_wake.retvorcity[i * w->numDimensions + j] = w->retvorcity[i][j];
        }
    }

    // states, ks and xs
    double* d_states;
    cudaMalloc(&d_states, sizeof(double) * w->size);
    cudaMemcpy(d_states, state_array, sizeof(double) * w->size, cudaMemcpyHostToDevice);

    double* rates;
    cudaMalloc(&rates, sizeof(double) * w->size);

    double tStart = TIME();
    for (size_t i = 1; i <= STEPS; i++) {
        OUT("\tStep", i);
        step_cuda(_dt, &cuda_wake, d_states, rates, w->size);
    }
    cudaDeviceSynchronize();
    double tEnd = TIME();
    OUT("Total Time (s)", tEnd - tStart);

    for (size_t i = 0; i < 20; i++) {
        std::cout << cuda_wake.position[i * 3] << " " << cuda_wake.position[i * 3 + 1] << " " << cuda_wake.position[i * 3 + 2] << " " << std::endl;
    }

    cudaFree(d_states);
    cudaFree(rates);
    cudaFree(cuda_wake.radius);
    cudaFree(cuda_wake.volume);
    cudaFree(cuda_wake.birthstrength);
    cudaFree(cuda_wake.position);
    cudaFree(cuda_wake.velocity);
    cudaFree(cuda_wake.vorticity);
    cudaFree(cuda_wake.retvorcity);
}