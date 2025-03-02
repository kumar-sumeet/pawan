#pragma once

#include "interaction/gpu_common.cuh"
#include "integration.h"
#include "interaction/gpu.cuh"
#include "wake/wake.h"
#include "interaction/interaction_utils_gpu.cuh"

namespace pawan{
    template<int threadBlockSize = 128, int unrollFactor = 1>
    class gpu_int : public __integration{

    private:
        void writePartSZL(void *fileHandle, double *p, double &t, int &numberOfParticles);
        void writePartDivFreeSZL(void *fileHandle,const double *divfreevorarr, const double *p,const double &t,const int &numberOfParticles);
        void writeGridSolSZL(void *fileHandle, double *p, double &t, size_t &stepnum, const int &nodes,
        const int &xdim, const int &ydim, const int &zdim);

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
__global__ void boundVorVindKernel(double4 *particles, const double3 *astpos, const size_t Npart, const double* circ,
                                   const size_t NbOfLfnLines, const int *NbOfAst, const double dt) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    double3 vb = {0,0,0}, Si, rast12, rast1r, licap, sicap, rast2r, vbi;
    double3 r;
    double t,span,simag,alpha,beta;
    double gamma0, gamma1;
    double factor;
    if(tid < Npart) {
        r.x = particles[2 * tid].x;
        r.y = particles[2 * tid].y;
        r.z = particles[2 * tid].z;

        int astidx = 0;
        for (size_t ilfn = 0; ilfn < NbOfLfnLines; ++ilfn) {
            for (size_t iast = 0; iast < NbOfAst[ilfn] - 2; ++iast) {
                Si = {0,0,0};
                licap={0,0,0};

                rast12 = astpos[astidx+1];
                subtract(rast12,astpos[astidx]);
                span = dnrm2(rast12);

                rast1r = r;
                subtract(rast1r,astpos[astidx]);
                t = dot_product(rast1r, rast12);
                t = t / (span * span);

                scaleadd(Si,rast12,t);
                add(Si, astpos[astidx]);

                scaleadd(licap, rast12, 1/span);

                sicap = r;
                subtract(sicap, Si);
                simag = dnrm2(sicap);
                scale(1/simag, sicap);

                rast2r = r;
                subtract(rast2r, astpos[astidx+1]);
                alpha = dot_product(rast2r, rast12);
                alpha = acos(-alpha / dnrm2(rast2r) / span);
                beta = dot_product(rast1r, rast12);
                beta = acos(beta / dnrm2(rast1r) / span);

                //t can be -ve
                gamma0 = (circ[astidx] * fabs(span * (1 - t)) + circ[astidx + 1] * fabs(t * span)) / (span * span);
                gamma1 = (circ[astidx + 1] - circ[astidx]) / span;
                cross(licap, sicap, vbi);
                scale(0.25 * M_1_PI / simag, vbi);
                factor = span * gamma0 * (cos(beta) + cos(alpha)) + simag * gamma1 * (sin(beta) - sin(alpha));
                scale(factor, vbi);

                add(vb, vbi);

                astidx++;
            }
        }
        particles[2 * tid].x += vb.x * dt;
        particles[2 * tid].y += vb.y * dt;
        particles[2 * tid].z += vb.z * dt;
    }
}

template<int threadBlockSize = 128, int unrollFactor = 1>
__global__ void boundVorStretchKernel(double4 *particles, const double3 *astpos, const size_t Npart, const double* circ,
                                   const size_t NbOfLfnLines, const int *NbOfAst, const double dt) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    double3 da = {0,0,0}, Si, rast12, rast1r, licap, sicap, rast2r, d1cap, d2cap, da1, da21, da22, da23, da24, licrosssi;
    double3 r, a;
    double t,span,simag,alpha,beta, d1mag, d2mag, dotprod;
    double gamma0, gamma1;
    double factor1, factor21, factor22, factor23, factor24;
    if(tid < Npart) {
        r.x = particles[2 * tid].x;
        r.y = particles[2 * tid].y;
        r.z = particles[2 * tid].z;
        a.x = particles[2 * tid + 1].x;
        a.y = particles[2 * tid + 1].y;
        a.z = particles[2 * tid + 1].z;

        int astidx = 0;
        for (size_t ilfn = 0; ilfn < NbOfLfnLines; ++ilfn) {
            for (size_t iast = 0; iast < NbOfAst[ilfn] - 2; ++iast) {
                Si = {0,0,0};
                licap={0,0,0};

                rast12 = astpos[astidx+1];
                subtract(rast12,astpos[astidx]);
                span = dnrm2(rast12);

                rast1r = r;
                subtract(rast1r,astpos[astidx]);
                t = dot_product(rast1r, rast12);
                t = t / (span * span);

                scaleadd(Si,rast12,t);
                add(Si, astpos[astidx]);

                scaleadd(licap, rast12, 1/span);

                sicap = r;
                subtract(sicap, Si);
                simag = dnrm2(sicap);
                scale(1/simag, sicap);

                rast2r = r;
                subtract(rast2r, astpos[astidx+1]);
                alpha = dot_product(rast2r, rast12);
                alpha = acos(-alpha / dnrm2(rast2r) / span);
                beta = dot_product(rast1r, rast12);
                beta = acos(beta / dnrm2(rast1r) / span);

                //t can be -ve
                gamma0 = (circ[astidx] * fabs(span * (1 - t)) + circ[astidx + 1] * fabs(t * span)) / (span * span);
                gamma1 = (circ[astidx + 1] - circ[astidx]) / span;

                d1cap = rast1r;
                d2cap = rast2r;
                d2mag = dnrm2(rast2r);
                d1mag = dnrm2(rast1r);
                scale(1/d2mag, d2cap);
                scale(1/d1mag, d1cap);

                cross(licap, a, da1);
                factor1 = span * gamma0 * (cos(beta) + cos(alpha)) + simag * gamma1 * (sin(beta) - sin(alpha));
                scale(factor1, da1);

                cross(licap, sicap, licrosssi);
                dotprod = dot_product(a, licrosssi);

                da21 = licap;
                factor21 = span * gamma0 * (sin(beta) - sin(alpha)) - simag * gamma1 * (cos(beta) + cos(alpha));
                scale(factor21, da21);

                da22=sicap;
                factor22 = -(2 * span * gamma0 * (cos(beta) + cos(alpha)) -
                             simag * gamma1 * (sin(beta) - sin(alpha)));
                scale(factor22, da22);

                da23 = d1cap;
                factor23 = -(span * gamma0 * sin(beta) - simag * gamma1 * cos(beta)) * cos(beta);
                scale(factor23, da23);

                da24 = d2cap;
                factor24 = -(span * gamma0 * sin(alpha) + simag * gamma1 * cos(alpha)) * cos(alpha);
                scale(factor24, da24);

                add(da21, da22);
                add(da21, da23);
                add(da21, da24);
                scale(dotprod, da21);
                add(da1, da21);
                scale(0.25 * M_1_PI / (simag * simag), da1);
                add(da, da1);

                astidx++;
            }
        }
        particles[2 * tid + 1].x += da.x * dt;
        particles[2 * tid + 1].y += da.y * dt;
        particles[2 * tid + 1].z += da.z * dt;
    }
}


template<int threadBlockSize = 128, int unrollFactor = 1>
__global__ void stepKernel(const double4 *source, double4 *target, double3 *rates, int *age, const size_t N, const double nu,const double dt) {

    int ownage = 0;
    double4 ownPosition, ownVorticity;
    double3 ownVelocity = {0,0,0}, ownRetVorticity = {0,0,0};

    size_t index = blockIdx.x * threadBlockSize + threadIdx.x;

    //cache own particle if index in bounds
    if(index < N){
        ownage = age[index];
        ownPosition = source[2 * index];
        ownVorticity = source[2 * index + 1];
    }

    interact_with_all<threadBlockSize,unrollFactor>(source, ownage, N, nu, ownPosition, ownVorticity, index, ownVelocity, ownRetVorticity);

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
__global__ void rk4stepKernel(const double4 *source, const double4 *x, double4 *target, double3 *rates, int *age, const size_t N, const double nu,const double dt) {

    int ownage = 0;
    double4 xPosition, xVorticity;
    double4 ownPosition, ownVorticity;
    double3 xVelocity = {0,0,0}, xRetVorticity = {0,0,0};

    size_t index = blockIdx.x * threadBlockSize + threadIdx.x;

    //cache own particle if index in bounds
    if(index < N){
        ownage = age[index];
        xPosition = x[2 * index];
        xVorticity = x[2 * index + 1];
        ownPosition = source[2 * index];
        ownVorticity = source[2 * index + 1];
    }

    interact_with_all<threadBlockSize,unrollFactor>(x, ownage, N, nu, xPosition, xVorticity, index, xVelocity, xRetVorticity);

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
void rk4Step(const double4 *source, double4 *target, int *age, const size_t N, const double nu, const double dt,
             double4* x1, double4* x2, double4* x3,
             double3* k1, double3* k2, double3* k3, double3* k4,
             cudaStream_t stream, const size_t threadBlocks) {

    // k1 = f(x,t)
    // x1 = x + 0.5*dt*k1
    stepKernel<threadBlockSize,unrollFactor><<<threadBlocks, threadBlockSize,0,stream>>>(source,x1,k1,age,N, nu, 0.5*dt);

    // k2 = f(x1, t+0.5*dt)
    // x2 = x + 0.5*dt*k2
    rk4stepKernel<threadBlockSize,unrollFactor><<<threadBlocks, threadBlockSize,0,stream>>>(source,x1,x2,k2,age,N, nu, 0.5*dt);

    // k3 = f(x2, t+0.5*dt)
    // x3 = x + dt*k3
    rk4stepKernel<threadBlockSize,unrollFactor><<<threadBlocks, threadBlockSize,0,stream>>>(source,x2,x3,k3,age,N, nu, dt);

    // k4 = f(x3, t+dt)
    //x2 used as dummy input since it is no longer required
    rk4stepKernel<threadBlockSize,unrollFactor><<<threadBlocks, threadBlockSize,0,stream>>>(source,x3,x2,k4,age,N, nu, dt);

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

template<int threadBlockSize = 128, int unrollFactor = 1>
__global__ void divfreevorKernel(const double4 *target, double3 *divfreevor, const size_t N) {
    double4 ownPosition;
    double3 divfreeomega_p {0,0,0};

    size_t index = blockIdx.x * threadBlockSize + threadIdx.x;

    //cache own particle if index in bounds
    if(index < N){
        ownPosition = target[2 * index];
    }

    divfreeomega_contrib_all<threadBlockSize,unrollFactor>(target, N, ownPosition, index, divfreeomega_p);

    if(index < N) {//alpha_p(div-free) = vol * omega
        divfreevor[index].x = divfreeomega_p.x * target[2 * index + 1].w;
        divfreevor[index].y = divfreeomega_p.y * target[2 * index + 1].w;
        divfreevor[index].z = divfreeomega_p.z * target[2 * index + 1].w;
    }
}
template<int threadBlockSize = 128, int unrollFactor = 1>
__global__ void relaxKernel(double4 *target,const double3 *divfreevor, const size_t N) {
    size_t tid = blockIdx.x * threadBlockSize + threadIdx.x;
    double factor = 0.3;   //f*delta_t from Pedrizetti's equation
    if(tid < N) {
        double alpha_p = dnrm2(target[2 * tid + 1]);
        double omega = dnrm2(divfreevor[tid]);
        target[2 * tid + 1].x = (1 - factor) * target[2 * tid + 1].x + factor * (alpha_p / omega) * divfreevor[tid].x;
        target[2 * tid + 1].y = (1 - factor) * target[2 * tid + 1].y + factor * (alpha_p / omega) * divfreevor[tid].y;
        target[2 * tid + 1].z = (1 - factor) * target[2 * tid + 1].z + factor * (alpha_p / omega) * divfreevor[tid].z;
    }
}
void divfreed3Todarr(double *divfreeArr,const double3 *divfreeD3,size_t N, size_t &stepnum){

    for(size_t i = 0; i < N; i++) {
        divfreeArr[       i] = divfreeD3[i].x;
        divfreeArr[   N + i] = divfreeD3[i].y;
        divfreeArr[ 2*N + i] = divfreeD3[i].z;
        divfreeArr[ 3*N + i] = sqrt(  divfreeD3[i].x * divfreeD3[i].x
                                    + divfreeD3[i].y * divfreeD3[i].y
                                    + divfreeD3[i].z * divfreeD3[i].z);
    }
}








template<int threadBlockSize = 128, int unrollFactor = 1>
__global__ void gridSolKernel(double3 *gridSol, const double4 *particles, const size_t Npart, const size_t Nnodes) {
    double3 nodePos;
    double3 nodeVel {0,0,0};
    double3 nodeVor {0,0,0};
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < Nnodes) {
        nodePos = gridSol[4 * tid];
        gridSol_contrib_all<threadBlockSize, unrollFactor>(particles, Npart, nodePos, nodeVel, nodeVor);
    }
    if(tid < Nnodes) {
        gridSol[4*tid + 1].x = nodeVel.x;
        gridSol[4*tid + 1].y = nodeVel.y;
        gridSol[4*tid + 1].z = nodeVel.z;
        gridSol[4*tid + 2].x = nodeVor.x;
        gridSol[4*tid + 2].y = nodeVor.y;
        gridSol[4*tid + 2].z = nodeVor.z;
        gridSol[4*tid + 3].x = dnrm2(nodeVor);
        //gridSol[4*tid + 2].y = ;
        //gridSol[4*tid + 2].z = ;
    }
}
template<int threadBlockSize = 128, int unrollFactor = 1>
__global__ void inflowEvalKernel(double *lambda, const double3 *airStaPositions, const double4 *particles, const size_t Npart, const size_t NairSta) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < NairSta) {
        double3 airStaPos = airStaPositions[tid];
        inflow_red<threadBlockSize, unrollFactor>(particles, Npart, airStaPos, lambda+3*tid);
    }
}
template<int threadBlockSize = 128, int unrollFactor = 1>
__global__ void inflowEvalKernelRed(double *lambda, const double3 airStaPosSingle, const double4 *particles, const size_t Npart) {
    inflow_red<threadBlockSize,unrollFactor>(particles, Npart, airStaPosSingle, lambda);
}
template<int threadBlockSize = 128, int unrollFactor = 1>
__global__ void updateVinfKernel(double4 *particles, const size_t Npart, const double3 *Vinf, const double dt) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < Npart) {
        scaleadd(particles[2*tid],Vinf[0],dt);
    }
}
void initialisegrid(double3 *gridSolD3, const int &nodes,
                    const double &xmin, const double &xmax,
                    const double &ymin, const double &ymax,
                    const double &zmin, const double &zmax,
                    const int &xdim, const int &ydim, const int &zdim){
    double xdelta = (xmax-xmin)/xdim;
    double ydelta = (ymax-ymin)/ydim;
    double zdelta = (zmax-zmin)/zdim;
    int index;
    for (int i = 0; i < xdim; ++i) {
        for (int j = 0; j < ydim; ++j) {
            for (int k = 0; k < zdim; ++k) {
                index = (k * ydim + j) * xdim + i;
                gridSolD3[4*index    ].x = xmin + i*xdelta;
                gridSolD3[4*index    ].y = ymin + j*ydelta;
                gridSolD3[4*index    ].z = zmin + k*zdelta;
                gridSolD3[4*index + 1].x = 0.0;
                gridSolD3[4*index + 1].y = 0.0;
                gridSolD3[4*index + 1].z = 0.0;
                gridSolD3[4*index + 2].x = 0.0;
                gridSolD3[4*index + 2].y = 0.0;
                gridSolD3[4*index + 2].z = 0.0;
                gridSolD3[4*index + 3].x = 0.0;
                gridSolD3[4*index + 3].y = 0.0;
                gridSolD3[4*index + 3].z = 0.0;
            }
        }
    }
}
void gridSold3Todarr(double *gridSolArr,double3 *gridSolD3,size_t nodes, size_t &stepnum){

    for(size_t i = 0; i < nodes; i++) {
        if(stepnum==0) {
            gridSolArr[i]             = gridSolD3[4*i].x;
            gridSolArr[nodes + i]     = gridSolD3[4*i].y;
            gridSolArr[2 * nodes + i] = gridSolD3[4*i].z;
        }
        gridSolArr[ 3*nodes + i] = gridSolD3[4*i + 1].x;
        gridSolArr[ 4*nodes + i] = gridSolD3[4*i + 1].y;
        gridSolArr[ 5*nodes + i] = gridSolD3[4*i + 1].z;
        gridSolArr[ 6*nodes + i] = gridSolD3[4*i + 2].x;
        gridSolArr[ 7*nodes + i] = gridSolD3[4*i + 2].y;
        gridSolArr[ 8*nodes + i] = gridSolD3[4*i + 2].z;
        gridSolArr[ 9*nodes + i] = gridSolD3[4*i + 3].x;
        gridSolArr[10*nodes + i] = gridSolD3[4*i + 3].y;
        gridSolArr[11*nodes + i] = gridSolD3[4*i + 3].z;
    }
}

template<int threadBlockSize, int unrollFactor>
void pawan::gpu_int<threadBlockSize,unrollFactor>::writePartSZL(void *fileHandle, double *p, double &t, int &numberOfParticles){
    int32_t zoneHandle;
    int varTypes[9] = {FieldDataType_Float,FieldDataType_Float,FieldDataType_Float,FieldDataType_Float,
                         FieldDataType_Float,FieldDataType_Float,FieldDataType_Float,FieldDataType_Float,
                         FieldDataType_Float}; //saving as float to minimise file size
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

    int32_t numZonesToRetain = 1;
    int32_t zonesToRetain[] = {1};
    tmp = tecFileWriterFlush(fileHandle, numZonesToRetain, zonesToRetain);
}

template<int threadBlockSize, int unrollFactor>
void pawan::gpu_int<threadBlockSize,unrollFactor>::writePartDivFreeSZL(void *fileHandle,const double *divfreevorarr, const double *p,const double &t,const int &numberOfParticles){
    int32_t zoneHandle;
    int varTypes[8] = {FieldDataType_Float,FieldDataType_Float,FieldDataType_Float,
                         FieldDataType_Float,FieldDataType_Float,FieldDataType_Float,
                         FieldDataType_Float, FieldDataType_Float}; //saving as float to minimise file size
    std::string s;
    std::stringstream convert;
    convert << t;
    s = convert.str();
    std::vector<int32_t> valueLocations(8, 1);
    int32_t tmp;
    tmp = tecZoneCreateIJK(fileHandle, s.c_str(), numberOfParticles, 1, 1, &varTypes[0], 0,&valueLocations[0], 0, 0, 0, 0, &zoneHandle);
    tmp = tecZoneVarWriteDoubleValues(fileHandle, zoneHandle, 1, 0, numberOfParticles, &p[0]);
    tmp = tecZoneVarWriteDoubleValues(fileHandle, zoneHandle, 2, 0, numberOfParticles, &p[numberOfParticles]);
    tmp = tecZoneVarWriteDoubleValues(fileHandle, zoneHandle, 3, 0, numberOfParticles, &p[2*numberOfParticles]);
    tmp = tecZoneVarWriteDoubleValues(fileHandle, zoneHandle, 4, 0, numberOfParticles, &divfreevorarr[0]);
    tmp = tecZoneVarWriteDoubleValues(fileHandle, zoneHandle, 5, 0, numberOfParticles, &divfreevorarr[numberOfParticles]);
    tmp = tecZoneVarWriteDoubleValues(fileHandle, zoneHandle, 6, 0, numberOfParticles, &divfreevorarr[2*numberOfParticles]);
    tmp = tecZoneVarWriteDoubleValues(fileHandle, zoneHandle, 7, 0, numberOfParticles, &divfreevorarr[3*numberOfParticles]);
    tmp = tecZoneVarWriteDoubleValues(fileHandle, zoneHandle, 8, 0, numberOfParticles, &p[3*numberOfParticles]);        //radius
    tmp =   tecZoneSetUnsteadyOptions(fileHandle, zoneHandle, t, 1);
    int32_t numZonesToRetain = 1;
    int32_t zonesToRetain[] = {1};
    tmp = tecFileWriterFlush(fileHandle, numZonesToRetain, zonesToRetain);
}

template<int threadBlockSize, int unrollFactor>
void pawan::gpu_int<threadBlockSize,unrollFactor>::writeGridSolSZL(void *fileHandle, double *gridSolArr, double &t, size_t &stepnum,const int &nodes,
                                                                   const int &xdim, const int &ydim, const int &zdim){
    int32_t zoneHandle;
    int varTypes[12] = {FieldDataType_Float,FieldDataType_Float,FieldDataType_Float,FieldDataType_Float,
                          FieldDataType_Float,FieldDataType_Float,FieldDataType_Float,FieldDataType_Float,
                          FieldDataType_Float,FieldDataType_Float,FieldDataType_Float,FieldDataType_Float};
    int shareVarFromZone[12] = {1,1,1,0,0,0,0,0,0,0,0,0};
    int32_t* shareVarPtr = (stepnum == 0 ? 0 : shareVarFromZone);
    std::string s;
    std::stringstream convert;
    convert << t;
    s = convert.str();
    std::vector<int32_t> valueLocations(12, 1);
    int32_t tmp;
    tmp = tecZoneCreateIJK(fileHandle, s.c_str(), xdim, ydim, zdim, &varTypes[0], shareVarPtr,&valueLocations[0], 0, 0, 0, 0, &zoneHandle);
    if(stepnum==0) {
        tmp = tecZoneVarWriteDoubleValues(fileHandle, zoneHandle, 1, 0, xdim*ydim*zdim, &gridSolArr[0]);
        tmp = tecZoneVarWriteDoubleValues(fileHandle, zoneHandle, 2, 0, xdim*ydim*zdim, &gridSolArr[nodes]);
        tmp = tecZoneVarWriteDoubleValues(fileHandle, zoneHandle, 3, 0, xdim*ydim*zdim, &gridSolArr[2 * nodes]);
    }
    tmp = tecZoneVarWriteDoubleValues(fileHandle, zoneHandle, 4, 0, nodes, &gridSolArr[ 3*nodes]);
    tmp = tecZoneVarWriteDoubleValues(fileHandle, zoneHandle, 5, 0, nodes, &gridSolArr[ 4*nodes]);
    tmp = tecZoneVarWriteDoubleValues(fileHandle, zoneHandle, 6, 0, nodes, &gridSolArr[ 5*nodes]);
    tmp = tecZoneVarWriteDoubleValues(fileHandle, zoneHandle, 7, 0, nodes, &gridSolArr[ 6*nodes]);
    tmp = tecZoneVarWriteDoubleValues(fileHandle, zoneHandle, 8, 0, nodes, &gridSolArr[ 7*nodes]);
    tmp = tecZoneVarWriteDoubleValues(fileHandle, zoneHandle, 9, 0, nodes, &gridSolArr[ 8*nodes]);
    tmp = tecZoneVarWriteDoubleValues(fileHandle, zoneHandle,10, 0, nodes, &gridSolArr[ 9*nodes]);
    tmp = tecZoneVarWriteDoubleValues(fileHandle, zoneHandle,11, 0, nodes, &gridSolArr[10*nodes]);
    tmp = tecZoneVarWriteDoubleValues(fileHandle, zoneHandle,12, 0, nodes, &gridSolArr[11*nodes]);
    tmp =   tecZoneSetUnsteadyOptions(fileHandle, zoneHandle, t, 1);

    int32_t numZonesToRetain = 1;
    int32_t zonesToRetain[] = {1};
    tmp = tecFileWriterFlush(fileHandle, numZonesToRetain, zonesToRetain);
}

template<int threadBlockSize, int unrollFactor>
void pawan::gpu_int<threadBlockSize,unrollFactor>::integrate(pawan::__system *S, pawan::__io *IO,
                                                             NetworkInterfaceTCP<OPawanRecvData, OPawanSendData> *networkCommunicatorTest,
                                                             bool diagnose) {
    //Because openmp does not work in cuda files currently, we switch measurement system
    auto tStart = std::chrono::high_resolution_clock::now();
    OPawanRecvData opawanrecvdata; OPawanSendData opawansenddata;
    networkCommunicatorTest->getrecieveBuffer(opawanrecvdata);
    int xrelax = opawanrecvdata.xrelax;
    int xsavepart = opawanrecvdata.xsavepart;
    int xsavegrid = opawanrecvdata.xsavegrid;
    double deltat = opawanrecvdata.deltat;
    _t = opawanrecvdata.t;
    _dt = opawanrecvdata.deltat;
    printf("_dt = %f \n", _dt);

    //for time-integration
    cudaStream_t integrateStream; cudaStreamCreate(&integrateStream); //not useful right now
    int maxnumberOfParticles = S->totalmaxParticles();
    size_t part_memsizei = maxnumberOfParticles * sizeof(int);
    size_t part_memsized4 = maxnumberOfParticles * 2 * sizeof(double4);
    size_t part_memsized3 = maxnumberOfParticles * 2 * sizeof(double3);
    double4 *gpuSource, *gpuTarget, *cpuBuffer;
    int *cpuage, *gpuage;
    checkGPUError(cudaMallocHost(&cpuBuffer, part_memsized4));//pinned memory buffer on the cpu
    checkGPUError(cudaMalloc(&gpuSource, part_memsized4));
    checkGPUError(cudaMalloc(&gpuTarget, part_memsized4));
    cpuage = (int*) malloc(part_memsizei);
    checkGPUError(cudaMalloc(&gpuage, part_memsizei));
    double4 *x1, *x2, *x3;
    double3 *k1, *k2, *k3, *k4;
    checkGPUError(cudaMalloc(&x1, part_memsized4));checkGPUError(cudaMalloc(&x2, part_memsized4));checkGPUError(cudaMalloc(&x3, part_memsized4));
    checkGPUError(cudaMalloc(&k1, part_memsized3));checkGPUError(cudaMalloc(&k2, part_memsized3));
    checkGPUError(cudaMalloc(&k3, part_memsized3));checkGPUError(cudaMalloc(&k4, part_memsized3));
    double *totalDiag;
    checkGPUError(cudaMallocHost(&totalDiag, 20*sizeof(double)));
    int numberOfParticles;
    size_t partThreadBlocks;

    //divergence free vorticity at each particle location; write to SZL file
    double3 *divfreevor_gpu;
    checkGPUError(cudaMalloc(&divfreevor_gpu, maxnumberOfParticles * sizeof(double3)));
    std::string szlwakedivfreefilename = IO->getSzlWakeDivFreeFile();
    std::string wakedivfreeVar("x y z divfreevor_x divfreevor_y divfreevor_z divfreeVor_strength radius");//8 variables
    void* fileWakeDivFreeHandle;
    int32_t resWakeDivFree = tecFileWriterOpen(szlwakedivfreefilename.c_str(),"IJK Ordered Zone",wakedivfreeVar.c_str(),1,0,FieldDataType_Double,0,&fileWakeDivFreeHandle);
    double3 *divfreevor = (double3*) malloc(maxnumberOfParticles * sizeof(double3));
    double *divfreevorarr = (double*) malloc(maxnumberOfParticles * 4 * sizeof(double));

    //particle transport with flow
    double3 *Vinf_gpu;
    checkGPUError(cudaMalloc(&Vinf_gpu, sizeof(double3)));

    //inflow evaluation at each airsta
    int NbOfLfnLines = opawanrecvdata.NbOfLfnLines;
    int *NbOfAst = opawanrecvdata.NbOfAst;
    double *astpos, *circ;
    int totalAirSta=0;
    for (size_t ilfn = 0; ilfn < NbOfLfnLines; ++ilfn)
        totalAirSta += NbOfAst[ilfn];
    printf("totalAirSta = %d \n", totalAirSta);
    double *lambda;
    checkGPUError(cudaMallocHost(&lambda, totalAirSta * 3 *sizeof(double)));
    double3 *lambda_gpu, *airStaPos_gpu;
    double *circ_gpu, *totalAirSta_gpu;
    int *NbOfAst_gpu;
    checkGPUError(cudaMalloc(&lambda_gpu, totalAirSta*sizeof(double3)));
    checkGPUError(cudaMalloc(&airStaPos_gpu, totalAirSta*sizeof(double3)));
    checkGPUError(cudaMalloc(&circ_gpu, totalAirSta*sizeof(double)));
    checkGPUError(cudaMalloc(&totalAirSta_gpu, sizeof(double)));
    checkGPUError(cudaMemcpy(totalAirSta_gpu,&totalAirSta,sizeof(double),cudaMemcpyHostToDevice));
    checkGPUError(cudaMalloc(&NbOfAst_gpu, NbOfLfnLines*sizeof(int)));
    size_t airStaThreadBlocks = (totalAirSta + threadBlockSize - 1) / threadBlockSize;

    //particle SZL file
    std::string szlwakefilename = IO->getSzlWakeFile();
    std::string wakeVar("x y z vor_x vor_y vor_z radius vol Vor_strength");//9 variables
    void* fileWakeHandle;
    int32_t resWake = tecFileWriterOpen(szlwakefilename.c_str(),"IJK Ordered Zone",wakeVar.c_str(),1,0,FieldDataType_Double,0,&fileWakeHandle);
    double *pWake = (double*) malloc(maxnumberOfParticles * 9 * sizeof(double));

    //grid data SZL file
    int xdim = opawanrecvdata.griddisc[0], ydim = opawanrecvdata.griddisc[1], zdim = opawanrecvdata.griddisc[2];
    double xmin = opawanrecvdata.gridlims[0], xmax = opawanrecvdata.gridlims[1];
    double ymin = opawanrecvdata.gridlims[2], ymax = opawanrecvdata.gridlims[3];
    double zmin = opawanrecvdata.gridlims[4], zmax = opawanrecvdata.gridlims[5];
    std::string szlvelfilename = IO->getSzlSolFile();
    std::string gridSolVar("x y z vx vy vz wx wy wz wtotal dummy1 dummy2");//12 variables
    void* fileGridSolHandle;
    int32_t resGridSol = tecFileWriterOpen(szlvelfilename.c_str(),"IJK Ordered Zone",gridSolVar.c_str(),1,0,FieldDataType_Double,0,&fileGridSolHandle);
    size_t Nnodes = xdim*ydim*zdim;
    size_t gridSol_memsized3 = Nnodes * 4* sizeof(double3);
    double *gridSolarr = (double*) malloc(gridSol_memsized3);
    double3 *gridSol = (double3*) malloc(gridSol_memsized3);
    double3 *gridSol_gpu;  //stores x,y,z coordinate of grid and vx, vy, vz vel components at each node
    checkGPUError(cudaMalloc(&gridSol_gpu, gridSol_memsized3));
    initialisegrid(gridSol,Nnodes,xmin,xmax,ymin,ymax,zmin,zmax,xdim,ydim,zdim);
    checkGPUError(cudaMemcpy(gridSol_gpu,gridSol,gridSol_memsized3,cudaMemcpyHostToDevice));
    size_t gridThreadBlocks = (Nnodes + threadBlockSize - 1) / threadBlockSize;

    size_t stepnum = 0;
    while(_t <= opawanrecvdata.tfinal){
        OUT("\tTime",_t);
        OUT("\tStepNum",stepnum);
        printf("_dt = %f \n", _dt);
        astpos = opawanrecvdata.astpos;
        circ = opawanrecvdata.circ;
        checkGPUError(cudaMemcpy(airStaPos_gpu,astpos,totalAirSta*sizeof(double3),cudaMemcpyHostToDevice));
        checkGPUError(cudaMemcpy(circ_gpu,circ,totalAirSta*sizeof(double),cudaMemcpyHostToDevice));
        checkGPUError(cudaMemcpy(NbOfAst_gpu,NbOfAst,NbOfLfnLines*sizeof(int),cudaMemcpyHostToDevice));

        //if(opawanrecvdata.bndvorincl)
        //    S->updateBoundVorEffect(&opawanrecvdata,_dt,stepnum);
        S->getParticles(reinterpret_cast<double *>(cpuBuffer),cpuage,stepnum);
        checkGPUError(cudaMemcpy(gpuSource,cpuBuffer,part_memsized4,cudaMemcpyHostToDevice));
        checkGPUError(cudaMemcpy(gpuage,cpuage,part_memsizei,cudaMemcpyHostToDevice));
        numberOfParticles = S->amountParticles();
        partThreadBlocks = (numberOfParticles + threadBlockSize - 1) / threadBlockSize;
        printf("\tnumberOfParticles = %10d \n",numberOfParticles);
        int astidx=0;
        double3 airStaPosSingle;
        for (size_t ilfn = 0; ilfn < NbOfLfnLines; ++ilfn) {
            for (size_t iast = 0; iast < NbOfAst[ilfn]; ++iast) {
                airStaPosSingle.x = astpos[astidx*3 ];
                airStaPosSingle.y = astpos[astidx*3 +1];
                airStaPosSingle.z = astpos[astidx*3 +2];
                lambda[astidx*3] = 0; lambda[astidx*3 +1] = 0; lambda[astidx*3 +2] = 0;
                inflowEvalKernelRed<threadBlockSize, unrollFactor><<<partThreadBlocks, threadBlockSize, 2*threadBlockSize*sizeof(double3), integrateStream >>>(lambda+astidx*3, airStaPosSingle, gpuSource, numberOfParticles);
                astidx++;
            }
        }
        cudaDeviceSynchronize();
        astidx = 0;
        for (size_t ilfn = 0; ilfn < NbOfLfnLines; ++ilfn) {
            for (size_t iast = 0; iast < NbOfAst[ilfn]; ++iast) {
                opawansenddata.lambda[astidx * 3    ] = lambda[astidx * 3    ];  //comment this for elliptical wing case
                opawansenddata.lambda[astidx * 3 + 1] = lambda[astidx * 3 + 1];
                opawansenddata.lambda[astidx * 3 + 2] = lambda[astidx * 3 + 2];
                if(opawanrecvdata.bndvorbndvorint) {    //bound vortex-bound vortex effect included (only needed for UT Austin rotor)
                    S->updateBoundVorBoundVorEffectVind(&opawanrecvdata, &opawansenddata, astidx, ilfn);
                }
                //printf("lambda       = %+10.5e, %+10.5e, %+10.5e \n",
                //       opawansenddata.lambda[astidx * 3], opawansenddata.lambda[astidx * 3 + 1], opawansenddata.lambda[astidx * 3 + 2]);
                astidx++;
            }
        }

        //stepKernel<threadBlockSize, unrollFactor><<<partThreadBlocks, threadBlockSize, 0, integrateStream >>>(gpuSource, gpuTarget, k1,numberOfParticles,S->getNu(), _dt);//k1 dummy here
        rk4Step<threadBlockSize, unrollFactor>(gpuSource,gpuTarget,gpuage,numberOfParticles,S->getNu(), _dt, x1, x2, x3, k1, k2, k3, k4, integrateStream, partThreadBlocks);
        if(opawanrecvdata.relax) {
            if (stepnum % xrelax == 0 && stepnum > 0) {
                printf("relax!!!-----------------\n");
                divfreevorKernel<threadBlockSize, unrollFactor><<<partThreadBlocks, threadBlockSize, 0, integrateStream >>>(
                        gpuTarget, divfreevor_gpu, numberOfParticles);
                relaxKernel<threadBlockSize, unrollFactor><<<partThreadBlocks, threadBlockSize, 0, integrateStream >>>(
                        gpuTarget, divfreevor_gpu, numberOfParticles);
                /*checkGPUError(cudaMemcpy(divfreevor,divfreevor_gpu,maxnumberOfParticles * sizeof(double3),cudaMemcpyDeviceToHost));
                divfreed3Todarr(divfreevorarr,divfreevor,numberOfParticles,stepnum);
                writePartDivFreeSZL(fileWakeDivFreeHandle,divfreevorarr,pWake,_t,numberOfParticles);
           */
            }
        }
        int transient_steps = opawanrecvdata.transientsteps;
        if (stepnum < transient_steps) {
            printf("Vinf = %3.2e, %3.2e, %3.2e \n", opawanrecvdata.Vinf[0], opawanrecvdata.Vinf[1], opawanrecvdata.Vinf[2]);
            opawanrecvdata.Vinf[2] = opawanrecvdata.Vinf[2] + 10.0 * (transient_steps - stepnum) / transient_steps;
            printf("Vinf + suppress = %3.2e, %3.2e, %3.2e", opawanrecvdata.Vinf[0], opawanrecvdata.Vinf[1],
                   opawanrecvdata.Vinf[2]);
        }

        if(opawanrecvdata.bndvorincl) {
            printf("incl bound vor vind effect ---------------- \n");
            boundVorVindKernel<threadBlockSize, unrollFactor><<<partThreadBlocks, threadBlockSize, 0, integrateStream >>>(gpuTarget, airStaPos_gpu, numberOfParticles, circ_gpu,
                                                                                                                          NbOfLfnLines, NbOfAst_gpu, _dt);
            /*printf("incl bound vor stretch effect ---------------- \n");
            boundVorStretchKernel<threadBlockSize, unrollFactor><<<partThreadBlocks, threadBlockSize, 0, integrateStream >>>(gpuTarget, airStaPos_gpu, numberOfParticles, circ_gpu,
                                                                                                                          NbOfLfnLines, NbOfAst_gpu, _dt);
            */cudaDeviceSynchronize();
        }

        printf("update Vinf-------- \n");
        checkGPUError(cudaMemcpy(Vinf_gpu,opawanrecvdata.Vinf,sizeof(double3),cudaMemcpyHostToDevice));
        updateVinfKernel<threadBlockSize, unrollFactor><<<partThreadBlocks, threadBlockSize, 0, integrateStream >>>(gpuTarget, numberOfParticles, Vinf_gpu, _dt);
        checkGPUError(cudaMemcpy(cpuBuffer, gpuTarget, part_memsized4, cudaMemcpyDeviceToHost));
        S->setParticles(reinterpret_cast<double *>(cpuBuffer));

        //if(stepnum % xsavepart==0 || (opawanrecvdata.tfinal - 2*opawanrecvdata.deltat)<=_t) { //write every n time steps only (|| (opawanrecvdata.tfinal-_t)/_dt<540)
        if(stepnum % xsavepart==0 ||  (opawanrecvdata.tfinal-_t)/_dt<=1250) { //write every n time steps only (|| (opawanrecvdata.tfinal-_t)/_dt<540)
            printf("writing particles szl file-------- \n");
            S->getParticles_arr(pWake);
            writePartSZL(fileWakeHandle, pWake, _t, numberOfParticles);
        }

        //if(stepnum % xsavegrid==0  || (opawanrecvdata.tfinal - 2*opawanrecvdata.deltat)<=_t) { //write every n time steps or every step for last revolution ( || (opawanrecvdata.tfinal-_t)/_dt<540)
        if(stepnum % xsavegrid==0  ||  (opawanrecvdata.tfinal-_t)/_dt<=1250) { //write every n time steps or every step for last revolution ( || (opawanrecvdata.tfinal-_t)/_dt<540)
            printf("writing gridsol szl file-------- \n");
            gridSolKernel<threadBlockSize, unrollFactor><<<gridThreadBlocks, threadBlockSize, 0, integrateStream >>>(
                    gridSol_gpu, gpuTarget, numberOfParticles, Nnodes);
            checkGPUError(cudaMemcpy(gridSol, gridSol_gpu, gridSol_memsized3, cudaMemcpyDeviceToHost));
            gridSold3Todarr(gridSolarr, gridSol, Nnodes, stepnum);
            writeGridSolSZL(fileGridSolHandle, gridSolarr, _t, stepnum, Nnodes, xdim, ydim, zdim);
        }
/*        if((opawanrecvdata.tfinal-_t)/_dt<3600){
            if(stepnum%9==-1) {
                printf("writing particles szl file & gridsol szl file-------- \n");
                S->getParticles_arr(pWake);
                writePartSZL(fileWakeHandle, pWake, _t, numberOfParticles);
                gridSolKernel<threadBlockSize, unrollFactor><<<gridThreadBlocks, threadBlockSize, 0, integrateStream >>>(
                        gridSol_gpu, gpuTarget, numberOfParticles, Nnodes);
                checkGPUError(cudaMemcpy(gridSol, gridSol_gpu, gridSol_memsized3, cudaMemcpyDeviceToHost));
                gridSold3Todarr(gridSolarr, gridSol, Nnodes, stepnum);
                writeGridSolSZL(fileGridSolHandle, gridSolarr, _t, stepnum, Nnodes, xdim, ydim, zdim);
            }
        }
*/
        S->split(stepnum);
        //S->merge(stepnum);
        networkCommunicatorTest->send_data(opawansenddata);
        stepnum = stepnum+1;
        if(_t <= (opawanrecvdata.tfinal - 1*opawanrecvdata.deltat)){
            networkCommunicatorTest->recieve_data(opawanrecvdata);
            S->addParticles(&opawanrecvdata, stepnum);
            _t = opawanrecvdata.t;
        }
        else
            break;
    }

    resWake = tecFileWriterClose(&fileWakeHandle);
    resWakeDivFree = tecFileWriterClose(&fileWakeDivFreeHandle);
    resGridSol = tecFileWriterClose(&fileGridSolHandle);

    checkGPUError(cudaFree(gpuSource));
    checkGPUError(cudaFree(gpuTarget));
    checkGPUError(cudaFreeHost(cpuBuffer));
    checkGPUError(cudaFree(x1));checkGPUError(cudaFree(x2));checkGPUError(cudaFree(x3));
    checkGPUError(cudaFree(k1));checkGPUError(cudaFree(k2));checkGPUError(cudaFree(k3));checkGPUError(cudaFree(k4));
    checkGPUError(cudaFreeHost(totalDiag));
    checkGPUError(cudaFree(divfreevor_gpu));free(divfreevor);free(divfreevorarr);
    checkGPUError(cudaFree(Vinf_gpu));
    checkGPUError(cudaFreeHost(lambda));checkGPUError(cudaFree(lambda_gpu));checkGPUError(cudaFree(airStaPos_gpu));
    free(pWake);
    free(gridSolarr);free(gridSol);checkGPUError(cudaFree(gridSol_gpu));

    auto tEnd = std::chrono::high_resolution_clock::now();
    OUT("Total Time (s)",std::chrono::duration<double>(tEnd - tStart).count());
}
template<int threadBlockSize, int unrollFactor>
void pawan::gpu_int<threadBlockSize,unrollFactor>::integrate(pawan::__system *S, pawan::__io *IO, bool diagnose) {
    FILE *f = IO->create_binary_file(".wake");
    FILE *fdiag = IO->create_binary_file(".diagnosis");
    double t = 0.0;
    fwrite(&t,sizeof(double),1,f);
    S->write(f);  //write particles info as is

    //Create two cuda streams so that integration and memory copies can happen at the same time
    cudaStream_t integrateStream;
    cudaStreamCreate(&integrateStream);

    //Memory allocations:
    //two GPU buffers so that in each step the result can be written
    //without having to wait on all threads finishing
    //one pinned memory buffer on the cpu for copying states back
    int numberOfParticles = S->amountParticles();
    int maxnumberOfParticles = S->totalmaxParticles();
    size_t part_memsizei = maxnumberOfParticles * sizeof(int);
    size_t mem_size = maxnumberOfParticles * 2 * sizeof(double4);
    size_t mem_sized3 = maxnumberOfParticles * 2 * sizeof(double3);
    double4 *gpuSource, *gpuTarget, *cpuBuffer;
    int *cpuage, *gpuage;
    double3 *rates;
    double3 *divfreevor;
    checkGPUError(cudaMallocHost(&cpuBuffer, mem_size));
    checkGPUError(cudaMalloc(&gpuSource, mem_size));
    checkGPUError(cudaMalloc(&gpuTarget, mem_size));
    checkGPUError(cudaMalloc(&rates, mem_sized3));
    checkGPUError(cudaMalloc(&divfreevor, maxnumberOfParticles * sizeof(double3)));
    cpuage = (int*) malloc(part_memsizei);
    checkGPUError(cudaMalloc(&gpuage, part_memsizei));
    double4 *x1, *x2, *x3;
    double3 *k1, *k2, *k3, *k4;
    checkGPUError(cudaMalloc(&x1, mem_size));checkGPUError(cudaMalloc(&x2, mem_size));checkGPUError(cudaMalloc(&x3, mem_size));
    checkGPUError(cudaMalloc(&k1, mem_sized3));checkGPUError(cudaMalloc(&k2, mem_sized3));checkGPUError(cudaMalloc(&k3, mem_sized3));checkGPUError(cudaMalloc(&k4, mem_sized3));
    double *totalDiag;
    checkGPUError(cudaMallocHost(&totalDiag, 20*sizeof(double)));

    //Transfer particles to GPU
    S->getParticles(reinterpret_cast<double *>(cpuBuffer),cpuage,0); //here particle age is not really relevant, it'll be zero in any case
    checkGPUError(cudaMemcpy(gpuSource,cpuBuffer,mem_size,cudaMemcpyHostToDevice));
    checkGPUError(cudaMemcpy(gpuage,cpuage,part_memsizei,cudaMemcpyHostToDevice));

    size_t threadBlocks = (numberOfParticles + threadBlockSize - 1) / threadBlockSize;
    std::string szlfilename = IO->getSzlWakeFile();
    std::string szldivfreefilename = IO->getSzlWakeDivFreeFile();
    std::string variables("x y z vor_x vor_y vor_z radius vol Vor_strength ");//9 variables
    std::string variablesDivFree("divfreeomega_x divfreeomega_y divfreeomega_z divfreevor_x divfreevor_y divfreevor_z ");//9 variables
    void *fileHandle, *fileHandleDivFree;
    int32_t res = tecFileWriterOpen(szlfilename.c_str(),"IJK Ordered Zone",variables.c_str(),1,0,FieldDataType_Double,0,&fileHandle);
    int32_t resDivFree = tecFileWriterOpen(szldivfreefilename.c_str(),"IJK Ordered Zone",variablesDivFree.c_str(),1,0,FieldDataType_Double,0,&fileHandleDivFree);
    double *p = (double*) malloc(maxnumberOfParticles * 9 * sizeof(double));

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
            //stepKernel<threadBlockSize, unrollFactor><<<threadBlocks, threadBlockSize, 0, integrateStream >>>(gpuSource, gpuTarget, rates,gpuage,numberOfParticles,S->getNu(), _dt);
            rk4Step<threadBlockSize, unrollFactor>(gpuSource,gpuTarget,gpuage,numberOfParticles,S->getNu(), _dt, x1, x2, x3, k1, k2, k3, k4, integrateStream, threadBlocks);
            divfreevorKernel<threadBlockSize, unrollFactor><<<threadBlocks, threadBlockSize, 0, integrateStream >>>(gpuTarget, divfreevor, numberOfParticles);
            relaxKernel<threadBlockSize, unrollFactor><<<threadBlocks, threadBlockSize, 0, integrateStream >>>(gpuTarget, divfreevor, numberOfParticles);
            cudaDeviceSynchronize();
        }

        //Start copy the result of the previous calculation
        checkGPUError(cudaMemcpy(cpuBuffer,gpuTarget,mem_size,cudaMemcpyDeviceToHost));

        //wait for memory copy to finish, then do all the things that need to be done on the cpu
        S->setParticles(reinterpret_cast<double *>(cpuBuffer));
        t = i*_dt; //The data is the one of the last step
        fwrite(&t,sizeof(double),1,f);
        S->write(f);  //write particles info after interaction of the last time step
        S->getParticles_arr(p);
        writePartSZL(fileHandle,p,t,numberOfParticles);
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
    fclose(fdiag);
    free(p);
    res = tecFileWriterClose(&fileHandle);
    resDivFree = tecFileWriterClose(&fileHandleDivFree);
    auto tEnd = std::chrono::high_resolution_clock::now();

    OUT("Total Time (s)",std::chrono::duration<double>(tEnd - tStart).count());

    checkGPUError(cudaFree(gpuSource));
    checkGPUError(cudaFree(gpuTarget));
    checkGPUError(cudaFreeHost(cpuBuffer));
    checkGPUError(cudaFreeHost(totalDiag));
    checkGPUError(cudaFree(divfreevor));
    checkGPUError(cudaFree(x1));checkGPUError(cudaFree(x2));checkGPUError(cudaFree(x3));
    checkGPUError(cudaFree(k1));checkGPUError(cudaFree(k2));checkGPUError(cudaFree(k3));checkGPUError(cudaFree(k4));
}
