
#include <cublas_v2.h>
#include <cuda.h>
#include "src/wake/wake_struct.h"

__device__ double euclidean_norm_cuda(const double* x, std::size_t n) {
    double sum_of_squares = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        sum_of_squares += x[i] * x[i];
    }
    return std::sqrt(sum_of_squares);
}

__device__ void KERNEL_CUDA_1(const double& rho,
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

__device__ void VELOCITY_CUDA_1(const double& kernel,
                                const double* vorticity,
                                const double* displacement,
                                double* velocity) {
    velocity[0] = vorticity[1] * displacement[2] - vorticity[2] * displacement[1];
    velocity[1] = vorticity[2] * displacement[0] - vorticity[0] * displacement[2];
    velocity[2] = vorticity[0] * displacement[1] - vorticity[1] * displacement[0];
    for (size_t i = 0; i < 3; i++)
        velocity[i] *= kernel;
};

__device__ void VORSTRETCH_CUDA_1(const double& q,
                                  const double& F,
                                  const double* source_vorticity,
                                  const double* target_vorticity,
                                  const double* displacement,
                                  double* retvorcity) {
    double* trgXsrc = new double[3];
    trgXsrc[0] = target_vorticity[1] * source_vorticity[2] - target_vorticity[2] * source_vorticity[1];
    trgXsrc[1] = target_vorticity[2] * source_vorticity[0] - target_vorticity[0] * source_vorticity[2];
    trgXsrc[2] = target_vorticity[0] * source_vorticity[1] - target_vorticity[1] * source_vorticity[0];

    double* crossed = new double[3];
    double* stretch = new double[3];

    for (size_t i = 0; i < 3; i++)
        crossed[i] = trgXsrc[i] * q;

    double roaxa = 0.0;
    for (size_t i = 0; i < 3; i++) {
        roaxa += displacement[i] * trgXsrc[i];
        stretch[i] = displacement[i] * F * roaxa;
    }

    for (size_t i = 0; i < 3; i++)
        retvorcity[i] += (crossed[i] + stretch[i]);

    delete[] trgXsrc;
    delete[] crossed;
    delete[] stretch;
};

__device__ void DIFFUSION_CUDA_1(const double& nu,
                                 const double& sigma,
                                 const double& Z,
                                 const double* source_vorticity,
                                 const double* target_vorticity,
                                 const double& source_volume,
                                 const double& target_volume,
                                 double* retvorcity) {
    double* va12 = new double[3];
    double* va21 = new double[3];
    double* dva = new double[3];

    for (size_t i = 0; i < 3; i++) {
        va12[i] = source_vorticity[i] * target_volume;
        va21[i] = target_vorticity[i] * source_volume;
    }

    double sig12 = 0.5 * sigma * sigma;
    for (size_t i = 0; i < 3; i++)
        dva[i] = (va12[i] - va21[i]) * (Z * nu / sig12);

    for (size_t i = 0; i < 3; i++)
        retvorcity[i] += dva[i];

    delete[] va12;
    delete[] va21;
    delete[] dva;
}

__device__ void INTERACT_CUDA_1(
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
    double* dr_target,
    double* da_source,
    double* da_target) {
    // kenerl computation
    double* displacement = new double[3];
    for (size_t i = 0; i < 3; i++)
        displacement[i] = r_target[i] - r_source[i];
    double rho = euclidean_norm_cuda(displacement, 3);
    double q = 0.0, F = 0.0, Z = 0.0;
    double sigma = std::sqrt(s_source * s_source + s_target * s_target) / 2.0;

    // velocity computation
    double* dr = new double[3];
    // target
    KERNEL_CUDA_1(rho, sigma, q, F, Z);
    VELOCITY_CUDA_1(q, a_source, displacement, dr);
    for (size_t i = 0; i < 3; i++)
        dr_target[i] += dr[i];

    // source
    VELOCITY_CUDA_1(-q, a_target, displacement, dr);
    for (size_t i = 0; i < 3; i++)
        dr_source[i] += dr[i];

    // Rate of change of vorticity computation
    double* da = new double[3];
    for (size_t i = 0; i < 3; i++)
        da[i] = 0.0;

    VORSTRETCH_CUDA_1(q, F, a_source, a_target, displacement, da);
    DIFFUSION_CUDA_1(nu, sigma, Z, a_source, a_target, v_source, v_target, da);

    // Target and source
    for (size_t i = 0; i < 3; i++) {
        da_target[i] += da[i];
        da_source[i] -= da[i];
    }
    delete[] dr;
    delete[] da;
    delete[] displacement;
}

__global__ void setStates_cuda_1(pawan::wake_cuda* w, const double* state) {
    size_t numDimensions = w->numDimensions;
    size_t np = w->size / 2 / numDimensions;
    size_t matrixsize = np * numDimensions;

    for (size_t i = 0; i < np; i++) {
        for (size_t j = 0; j < numDimensions; j++) {
            w->position[i * numDimensions + j] = state[i * numDimensions + j];
        }
    }

    for (size_t i = 0; i < np; i++) {
        for (size_t j = 0; j < numDimensions; j++) {
            w->vorticity[i * numDimensions + j] = state[i * numDimensions + j + matrixsize];
        }
    }
}

__global__ void getStates_cuda_1(pawan::wake_cuda* w, double* state) {
    size_t numDimensions = w->numDimensions;
    size_t numParticles = w->numParticles;
    for (size_t i = 0; i < numParticles; i++) {
        for (size_t j = 0; j < numDimensions; j++) {
            size_t ind = i * numDimensions + j;
            state[ind] = w->position[ind];
            ind += w->size / 2;
            state[ind] = w->vorticity[ind];
        }
    }
}

__global__ void getRates_cuda_1(pawan::wake_cuda* w, double* rate) {
    size_t numDimensions = w->numDimensions;
    size_t numParticles = w->numParticles;
    for (size_t i = 0; i < numParticles; i++) {
        for (size_t j = 0; j < numDimensions; j++) {
            size_t ind = i * numDimensions + j;
            rate[ind] = w->velocity[ind];
            ind += w->size / 2;
            rate[ind] = w->retvorcity[ind];
        }
    }
}

__global__ void interact_cuda_1(pawan::wake_cuda* w) {
    size_t numDimensions = w->numDimensions;
    size_t numParticles = w->numParticles;
    for (size_t i = 0; i < numParticles; i++)
        for (size_t j = 0; j < numDimensions; j++)
            w->velocity[i * numDimensions + j] = w->retvorcity[i * numDimensions + j] = 0.0;

    for (size_t i_src = 0; i_src < numParticles; i_src++) {
        double* r_src = new double[numDimensions];
        double* a_src = new double[numDimensions];
        double* dr_src = new double[numDimensions];
        double* da_src = new double[numDimensions];
        for (size_t j = 0; j < numDimensions; j++) {
            r_src[j] = w->position[i_src * numDimensions + j];
            a_src[j] = w->vorticity[i_src * numDimensions + j];
            dr_src[j] = w->velocity[i_src * numDimensions + j];
            da_src[j] = w->retvorcity[i_src * numDimensions + j];
        }
        double s_src = w->radius[i_src];
        double v_src = w->volume[i_src];
        for (size_t i_trg = i_src + 1; i_trg < numParticles; i_trg++) {
            double* r_trg = new double[numDimensions];
            double* a_trg = new double[numDimensions];
            double* dr_trg = new double[numDimensions];
            double* da_trg = new double[numDimensions];
            for (size_t j = 0; j < numDimensions; j++) {
                r_trg[j] = w->position[i_trg * numDimensions + j];
                a_trg[j] = w->vorticity[i_trg * numDimensions + j];
                dr_trg[j] = w->velocity[i_trg * numDimensions + j];
                da_trg[j] = w->retvorcity[i_trg * numDimensions + j];
            }
            double s_trg = w->radius[i_trg];
            double v_trg = w->volume[i_trg];

            INTERACT_CUDA_1(w->_nu, s_src, s_trg, r_src, r_trg, a_src, a_trg, v_src, v_trg, dr_src, dr_trg, da_src, da_trg);
            for (size_t j = 0; j < numDimensions; j++) {
                w->position[i_src * numDimensions + j] = r_src[j];
                w->vorticity[i_src * numDimensions + j] = a_src[j];
                w->velocity[i_src * numDimensions + j] = dr_src[j];
                w->retvorcity[i_src * numDimensions + j] = da_src[j];

                w->position[i_trg * numDimensions + j] = r_trg[j];
                w->vorticity[i_trg * numDimensions + j] = a_trg[j];
                w->velocity[i_trg * numDimensions + j] = dr_trg[j];
                w->retvorcity[i_trg * numDimensions + j] = da_trg[j];
            }
        }
    }
}

__global__ void rk4_cal(const double dt, double* x, const double* k, const double* d_states, const int len) {
    for (size_t i = 0; i < len; i++) {
        x[i] = k[i];
        x[i] *= 0.5 * dt;
        x[i] += d_states[i];
    }
}

__global__ void rk4_final(const double dt, double* d_states, double* k1, double* k2, const double* k3, const double* k4, const int len) {
    for (size_t i = 0; i < len; i++) {
        k1[i] += k4[i];
        k1[i] *= dt / 6.;

        k2[i] += k3[i];
        k2[i] *= dt / 3.;

        k1[i] += k2[i];

        d_states[i] += k1[i];
    }
}

extern "C" void step_cuda_1(const double dt, pawan::wake_struct* w, double* h_states, const int len) {
    // declear wake_cuda and malloc memory with cuda unified memory
    pawan::wake_cuda cuda_wake;
    cuda_wake.size = w->size;
    cuda_wake.numParticles = w->numParticles;
    cudaMallocManaged(&cuda_wake.position, sizeof(double) * len);
    cudaMallocManaged(&cuda_wake.velocity, sizeof(double) * len);
    cudaMallocManaged(&cuda_wake.vorticity, sizeof(double) * len);
    cudaMallocManaged(&cuda_wake.retvorcity, sizeof(double) * len);
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

    double* d_states;
    cudaMalloc(&d_states, sizeof(double) * len);
    cudaMemcpy(d_states, h_states, sizeof(double) * len, cudaMemcpyHostToDevice);

    double *x1, *x2, *x3, *k1, *k2, *k3, *k4;
    cudaMalloc(&x1, sizeof(double) * len);
    cudaMalloc(&x2, sizeof(double) * len);
    cudaMalloc(&x3, sizeof(double) * len);
    cudaMalloc(&k1, sizeof(double) * len);
    cudaMalloc(&k2, sizeof(double) * len);
    cudaMalloc(&k3, sizeof(double) * len);
    cudaMalloc(&k4, sizeof(double) * len);

    cudaMemcpy(x1, d_states, sizeof(double) * len, cudaMemcpyDeviceToDevice);

    // k1 = f(x,t)
    setStates_cuda_1<<<1, 1>>>(&cuda_wake, d_states);
    interact_cuda_1<<<1, 1>>>(&cuda_wake);
    getRates_cuda_1<<<1, 1>>>(&cuda_wake, k1);
    //  std::cout << "k1:        " << k1[0] << " " << k1[1] << " " << k1[2] << " " << std::endl;
    // x1 = x + 0.5*dt*k1
    rk4_cal<<<1, 1>>>(dt, x1, k1, d_states, len);

    // k2 = f(x1, t+0.5*dt)
    setStates_cuda_1<<<1, 1>>>(&cuda_wake, x1);
    interact_cuda_1<<<1, 1>>>(&cuda_wake);
    getRates_cuda_1<<<1, 1>>>(&cuda_wake, k2);
    //  std::cout << "k2:       " << k2[0] << " " << k2[1] << " " << k2[2] << " " << std::endl;
    // x2 = x1 + 0.5*dt*dx2
    rk4_cal<<<1, 1>>>(dt, x2, k2, d_states, len);
    //  std::cout << "x2:        " << x2[0] << " " << x2[1] << " " << x2[2] << " " << std::endl;

    // k3 = f(x2, t+0.5*dt)
    setStates_cuda_1<<<1, 1>>>(&cuda_wake, x2);
    interact_cuda_1<<<1, 1>>>(&cuda_wake);
    getRates_cuda_1<<<1, 1>>>(&cuda_wake, k3);
    //  std::cout << "velocity: " << w->velocity[0][0] << " " << w->velocity[0][1] << " " << w->velocity[0][2] << " " << std::endl;
    //  std::cout << "k3:       " << k3[0] << " " << k3[1] << " " << k3[2] << " " << std::endl;

    // x3 = x2 + dt*k3
    rk4_cal<<<1, 1>>>(dt, x3, k3, d_states, len);

    // k4 = f(x3, t+dt)
    setStates_cuda_1<<<1, 1>>>(&cuda_wake, x3);
    interact_cuda_1<<<1, 1>>>(&cuda_wake);
    getRates_cuda_1<<<1, 1>>>(&cuda_wake, k4);

    rk4_final <<<1, 1>>>(dt, d_states, k1, k2, k3, k4, len);

    setStates_cuda_1<<<1, 1>>>(&cuda_wake, d_states);

    cudaFree(cuda_wake.radius);
    cudaFree(cuda_wake.volume);
    cudaFree(cuda_wake.birthstrength);
    cudaFree(cuda_wake.position);
    cudaFree(cuda_wake.velocity);
    cudaFree(cuda_wake.vorticity);
    cudaFree(cuda_wake.retvorcity);
    cudaFree(d_states);
    cudaFree(x1);
    cudaFree(x2);
    cudaFree(x3);
    cudaFree(k1);
    cudaFree(k2);
    cudaFree(k3);
    cudaFree(k4);
}