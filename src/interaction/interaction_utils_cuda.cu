
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
    float3 trgXsrc = make_float3(target_vorticity[1] * source_vorticity[2] - target_vorticity[2] * source_vorticity[1],
                                 target_vorticity[1] * source_vorticity[2] - target_vorticity[2] * source_vorticity[1],
                                 target_vorticity[0] * source_vorticity[1] - target_vorticity[1] * source_vorticity[0]);
    double roaxa = 0.0;
    roaxa += displacement.x * trgXsrc.x;
    roaxa += displacement.y * trgXsrc.y;
    roaxa += displacement.z * trgXsrc.z;

    retvorcity.x += ((trgXsrc.x * q) + (displacement.x * F * roaxa));
    retvorcity.y += ((trgXsrc.y * q) + (displacement.y * F * roaxa));
    retvorcity.z += ((trgXsrc.z * q) + (displacement.z * F * roaxa));
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
    float3 displacement = make_float3(r_target[0] - r_source[0], r_target[1] - r_source[1], r_target[2] - r_source[2]);
    double rho = std::sqrt(displacement.x * displacement.x + displacement.y * displacement.y + displacement.z * displacement.z + SOFTENING);
    double q = 0.0, F = 0.0, Z = 0.0;
    double sigma = std::sqrt(s_source * s_source + s_target * s_target) / 2.0;

    float3 dr = make_float3(0.0, 0.0, 0.0);
    KERNEL_CUDA(rho, sigma, q, F, Z);
    VELOCITY_CUDA(q, a_source, displacement, dr);
    dr_source[0] += dr.x;
    dr_source[1] += dr.y;
    dr_source[2] += dr.z;

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
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < w.size) {
        w.velocity[tid] = w.retvorcity[tid] = 0.0;
    }
}

__global__ void interact_cuda(pawan::wake_cuda w) {
    int i_src = blockIdx.x * blockDim.x + threadIdx.x;
    size_t numDimensions = w.numDimensions;

    double *r_src, *a_src, *dr_src, *da_src;
    double s_src, v_src;
    if (i_src < w.numParticles) {
        r_src = &(w.position[i_src * numDimensions]);
        a_src = &(w.vorticity[i_src * numDimensions]);
        dr_src = &(w.velocity[i_src * numDimensions]);
        da_src = &(w.retvorcity[i_src * numDimensions]);
        s_src = w.radius[i_src];
        v_src = w.volume[i_src];
    }

    __shared__ double r_trgs[BLOCKSIZE * 3];
    __shared__ double a_trgs[BLOCKSIZE * 3];
    __shared__ double s_trgs[BLOCKSIZE];
    __shared__ double v_trgs[BLOCKSIZE];

    for (int tile = 0; tile < (w.numParticles + BLOCKSIZE - 1) / BLOCKSIZE; tile++) {
        int index = threadIdx.x + tile * BLOCKSIZE;

        if (index < w.numParticles) {
            r_trgs[threadIdx.x * numDimensions] = w.position[index * numDimensions];
            r_trgs[threadIdx.x * numDimensions + 1] = w.position[index * numDimensions + 1];
            r_trgs[threadIdx.x * numDimensions + 2] = w.position[index * numDimensions + 2];

            a_trgs[threadIdx.x * numDimensions] = w.vorticity[index * numDimensions];
            a_trgs[threadIdx.x * numDimensions + 1] = w.vorticity[index * numDimensions + 1];
            a_trgs[threadIdx.x * numDimensions + 2] = w.vorticity[index * numDimensions + 2];

            s_trgs[threadIdx.x] = w.radius[index];
            v_trgs[threadIdx.x] = w.volume[index];
        }

        __syncthreads();

        int num_targets = (BLOCKSIZE < w.numParticles - tile * BLOCKSIZE) ? BLOCKSIZE : w.numParticles - tile * BLOCKSIZE;
        if (i_src < w.numParticles) {
            for (size_t i_trg = 0; i_trg < num_targets; i_trg++) {
                const double* r_trg = &(r_trgs[i_trg * numDimensions]);
                const double* a_trg = &(a_trgs[i_trg * numDimensions]);
                double s_trg = s_trgs[i_trg];
                double v_trg = v_trgs[i_trg];

                INTERACT_CUDA(w._nu, s_src, s_trg, r_src, r_trg, a_src, a_trg, v_src, v_trg, dr_src, da_src);
            }
        }

        __syncthreads();
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

void step_cuda(const double dt, pawan::wake_cuda* w, double* d_states, double* x1, double* x2, double* x3, double* k1, double* k2, double* k3, double* k4, const int len) {
    cudaMemcpy(x1, d_states, sizeof(double) * len, cudaMemcpyDeviceToDevice);

    int numBlocks_states = (len / 2 + BLOCKSIZE - 1) / BLOCKSIZE;
    int numBlocks_rk = (len + BLOCKSIZE - 1) / BLOCKSIZE;
    int numBlocks_interact = (w->numParticles + BLOCKSIZE - 1) / BLOCKSIZE;

    //  k1 = f(x,t)
    setStates_cuda<<<numBlocks_states, BLOCKSIZE>>>(*w, d_states);
    clear<<<numBlocks_states, BLOCKSIZE>>>(*w);
    interact_cuda<<<numBlocks_interact, BLOCKSIZE>>>(*w);
    getRates_cuda<<<numBlocks_states, BLOCKSIZE>>>(*w, k1);

    // x1 = x + 0.5*dt*k1
    rk4_process<<<numBlocks_rk, BLOCKSIZE>>>(dt, x1, k1, d_states, FACTOR1, len);

    // k2 = f(x1, t+0.5*dt)
    setStates_cuda<<<numBlocks_states, BLOCKSIZE>>>(*w, x1);
    clear<<<numBlocks_states, BLOCKSIZE>>>(*w);
    interact_cuda<<<numBlocks_interact, BLOCKSIZE>>>(*w);
    getRates_cuda<<<numBlocks_states, BLOCKSIZE>>>(*w, k2);

    // x2 = x1 + 0.5*dt*dx2
    rk4_process<<<numBlocks_rk, BLOCKSIZE>>>(dt, x2, k2, d_states, FACTOR1, len);

    // k3 = f(x2, t+0.5*dt)
    setStates_cuda<<<numBlocks_states, BLOCKSIZE>>>(*w, x2);
    clear<<<numBlocks_states, BLOCKSIZE>>>(*w);
    interact_cuda<<<numBlocks_interact, BLOCKSIZE>>>(*w);
    getRates_cuda<<<numBlocks_states, BLOCKSIZE>>>(*w, k3);

    // x3 = x2 + dt*k3
    rk4_process<<<numBlocks_rk, BLOCKSIZE>>>(dt, x3, k3, d_states, FACTOR2, len);

    // k4 = f(x3, t+dt)
    setStates_cuda<<<numBlocks_states, BLOCKSIZE>>>(*w, x3);
    clear<<<numBlocks_states, BLOCKSIZE>>>(*w);
    interact_cuda<<<numBlocks_interact, BLOCKSIZE>>>(*w);
    getRates_cuda<<<numBlocks_states, BLOCKSIZE>>>(*w, k4);

    rk4_final<<<numBlocks_rk, BLOCKSIZE>>>(dt, d_states, k1, k2, k3, k4, len);

    setStates_cuda<<<numBlocks_states, BLOCKSIZE>>>(*w, d_states);
}

extern "C" void cuda_step_wrapper(const double _dt, pawan::__wake* h_wake, double* h_states) {
    pawan::wake_cuda d_wake;
    d_wake.size = h_wake->_size;
    d_wake.numDimensions = h_wake->_numDimensions;
    d_wake.numParticles = h_wake->_numParticles;
    cudaMallocManaged(&d_wake.position, sizeof(double) * h_wake->_size);
    cudaMallocManaged(&d_wake.velocity, sizeof(double) * h_wake->_size);
    cudaMallocManaged(&d_wake.vorticity, sizeof(double) * h_wake->_size);
    cudaMallocManaged(&d_wake.retvorcity, sizeof(double) * h_wake->_size);
    cudaMallocManaged(&d_wake.radius, sizeof(double) * h_wake->_numParticles);
    cudaMallocManaged(&d_wake.volume, sizeof(double) * h_wake->_numParticles);
    cudaMallocManaged(&d_wake.birthstrength, sizeof(double) * h_wake->_numParticles);

    for (size_t i = 0; i < h_wake->_numParticles; i++) {
        d_wake.radius[i] = gsl_vector_get(h_wake->_radius, i);
        d_wake.volume[i] = gsl_vector_get(h_wake->_volume, i);
        d_wake.birthstrength[i] = gsl_vector_get(h_wake->_birthstrength, i);
        for (size_t j = 0; j < h_wake->_numDimensions; j++) {
            d_wake.position[i * h_wake->_numDimensions + j] = gsl_matrix_get(h_wake->_position, i, j);
            d_wake.velocity[i * h_wake->_numDimensions + j] = gsl_matrix_get(h_wake->_velocity, i, j);
            d_wake.vorticity[i * h_wake->_numDimensions + j] = gsl_matrix_get(h_wake->_vorticity, i, j);
            d_wake.retvorcity[i * h_wake->_numDimensions + j] = gsl_matrix_get(h_wake->_retvorcity, i, j);
        }
    }

    double* d_states;
    cudaMalloc(&d_states, sizeof(double) * h_wake->_size);
    cudaMemcpy(d_states, h_states, sizeof(double) * h_wake->_size, cudaMemcpyHostToDevice);

    double *x1, *x2, *x3, *k1, *k2, *k3, *k4;
    cudaMalloc(&x1, sizeof(double) * h_wake->_size);
    cudaMalloc(&x2, sizeof(double) * h_wake->_size);
    cudaMalloc(&x3, sizeof(double) * h_wake->_size);
    cudaMalloc(&k1, sizeof(double) * h_wake->_size);
    cudaMalloc(&k2, sizeof(double) * h_wake->_size);
    cudaMalloc(&k3, sizeof(double) * h_wake->_size);
    cudaMalloc(&k4, sizeof(double) * h_wake->_size);

    double tStart = TIME();
    for (size_t i = 1; i <= STEPS; i++) {
        OUT("\tStep", i);
        step_cuda(_dt, &d_wake, d_states, x1, x2, x3, k1, k2, k3, k4, h_wake->_size);
    }
    cudaDeviceSynchronize();
    double tEnd = TIME();
    OUT("Total Time (s)", tEnd - tStart);

    for (size_t i = 0; i < h_wake->_numParticles; i++) {
        std::cout << d_wake.position[i * 3] << " " << d_wake.position[i * 3 + 1] << " " << d_wake.position[i * 3 + 2] << " " << std::endl;
    }

    cudaFree(d_states);
    cudaFree(x1);
    cudaFree(x2);
    cudaFree(x3);
    cudaFree(k1);
    cudaFree(k2);
    cudaFree(k3);
    cudaFree(k4);
    cudaFree(d_wake.radius);
    cudaFree(d_wake.volume);
    cudaFree(d_wake.birthstrength);
    cudaFree(d_wake.position);
    cudaFree(d_wake.velocity);
    cudaFree(d_wake.vorticity);
    cudaFree(d_wake.retvorcity);
}