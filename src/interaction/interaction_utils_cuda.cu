
#include <cublas_v2.h>
#include <cuda.h>
#include <stdio.h>
#include "src/integration/integration.h"
#include "src/io/io.h"
#include "src/utils/timing_utils.h"
#include "src/wake/wake_struct.h"

#define BLOCKSIZE 256
#define SOFTENING_CUDA 1e-12f
#define FACTOR1 0.5
#define FACTOR2 1.0

__device__ void euclidean_norm_cuda(double& res, const double* x, std::size_t n) {
    double sum_of_squares = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        sum_of_squares += x[i] * x[i];
    }
    res = std::sqrt(sum_of_squares + SOFTENING_CUDA);
}

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
                              const double* displacement,
                              double* velocity) {
    velocity[0] = (vorticity[1] * displacement[2] - vorticity[2] * displacement[1]) * kernel;
    velocity[1] = (vorticity[2] * displacement[0] - vorticity[0] * displacement[2]) * kernel;
    velocity[2] = (vorticity[0] * displacement[1] - vorticity[1] * displacement[0]) * kernel;
};

__device__ void VORSTRETCH_CUDA(const double& q,
                                const double& F,
                                const double* source_vorticity,
                                const double* target_vorticity,
                                const double* displacement,
                                double* retvorcity) {
    double* trgXsrc = (double*)malloc(sizeof(double) * 3);
    trgXsrc[0] = target_vorticity[1] * source_vorticity[2] - target_vorticity[2] * source_vorticity[1];
    trgXsrc[1] = target_vorticity[2] * source_vorticity[0] - target_vorticity[0] * source_vorticity[2];
    trgXsrc[2] = target_vorticity[0] * source_vorticity[1] - target_vorticity[1] * source_vorticity[0];

    double roaxa = 0.0;
    for (size_t i = 0; i < 3; i++) {
        roaxa += displacement[i] * trgXsrc[i];
    }

    for (size_t i = 0; i < 3; i++)
        retvorcity[i] += ((trgXsrc[i] * q) + (displacement[i] * F * roaxa));

    free(trgXsrc);
};

__device__ void DIFFUSION_CUDA(const double& nu,
                               const double& sigma,
                               const double& Z,
                               const double* source_vorticity,
                               const double* target_vorticity,
                               const double& source_volume,
                               const double& target_volume,
                               double* retvorcity) {
    double sig12 = 0.5 * sigma * sigma;
    for (size_t i = 0; i < 3; i++)
        retvorcity[i] += ((source_vorticity[i] * target_volume) - (target_vorticity[i] * source_volume)) * (Z * nu / sig12);
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
    // double* dr_source,
    double* dr_target,
    // double* da_source,
    double* da_target) {
    // kenerl computation
    double* displacement = (double*)malloc(sizeof(double) * 3);
    for (size_t i = 0; i < 3; i++)
        displacement[i] = r_target[i] - r_source[i];
    double rho;
    euclidean_norm_cuda(rho, displacement, 3);
    double q = 0.0, F = 0.0, Z = 0.0;
    double sigma = std::sqrt(s_source * s_source + s_target * s_target) / 2.0;

    // velocity computation
    double* dr = (double*)malloc(sizeof(double) * 3);

    for (size_t i = 0; i < 3; i++)
        dr[i] = 0.0;
    // target
    KERNEL_CUDA(rho, sigma, q, F, Z);
    VELOCITY_CUDA(q, a_source, displacement, dr);
    for (size_t i = 0; i < 3; i++)
        atomicAdd(dr_target + i, dr[i]);

    // Rate of change of vorticity computation
    double* da = (double*)malloc(sizeof(double) * 3);
    for (size_t i = 0; i < 3; i++)
        da[i] = 0.0;

    VORSTRETCH_CUDA(q, F, a_source, a_target, displacement, da);
    DIFFUSION_CUDA(nu, sigma, Z, a_source, a_target, v_source, v_target, da);

    // Target and source
    for (size_t i = 0; i < 3; i++)
        atomicAdd(da_target + i, da[i]);

    free(dr);
    free(da);
    free(displacement);
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

        double s_src = w.radius[i_src];
        double v_src = w.volume[i_src];

        for (size_t i_trg = 0; i_trg < w.numParticles; i_trg++) {
            const double* r_trg = &(w.position[i_trg * numDimensions]);
            const double* a_trg = &(w.vorticity[i_trg * numDimensions]);
            double* dr_trg = &(w.velocity[i_trg * numDimensions]);
            double* da_trg = &(w.retvorcity[i_trg * numDimensions]);
            double s_trg = w.radius[i_trg];
            double v_trg = w.volume[i_trg];

            INTERACT_CUDA(w._nu, s_src, s_trg, r_src, r_trg, a_src, a_trg, v_src, v_trg, dr_trg, da_trg);
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
    cudaDeviceSynchronize();
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
    double tEnd = TIME();
    OUT("Total Time (s)", tEnd - tStart);

    for (size_t i = 0; i < w->numParticles; i++) {
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