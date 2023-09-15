//
// Created by ge56beh on 11.09.23.
//

#ifndef PAWAN_GAUSSIAN_CUH
#define PAWAN_GAUSSIAN_CUH

__device__ inline double ZETASIG_GPU(const double &rho, const double &sigma);
__device__ inline double QSIG_GPU(const double &rho, const double &sigma);
__device__ inline void KERNEL_GPU(double rho, double sigma, double &q, double &f, double &z, double &n);
__device__ inline void ENST_GPU(const double &rho, const double &sigma, double &F1, double &F2);
__device__ inline void ENSTF_GPU(const double &rho, const double &sigma, double &F1);
__device__ inline double ENST_GPU(	const double &sigma);
__device__ inline double ENSTF_GPU(	const double &sigma);

__device__ inline double ZETASIG_GPU(	const double &rho, const double &sigma) {
    double rho_bar = rho/sigma;
    return exp(-0.5*rho_bar*rho_bar)/pow(sigma,3.0)/pow(2.0*M_PI,1.5);
}

__device__ double QSIG_GPU(const double &rho, const double &sigma) {
    double rho_bar = rho/sigma;
    double rho_bar2 = rho_bar * rho_bar;
    double sig3 = sigma*sigma*sigma;
    double Z = ZETASIG_GPU(rho,sigma);
    double phi = 0.25*M_1_PI*erf(M_SQRT1_2*rho_bar)/sig3;
    return (phi/rho_bar - Z)/rho_bar2;
}

__device__ inline void KERNEL_GPU(const double rho, const double sigma, double &q, double &F, double &Z, double &n) {
    Z = ZETASIG_GPU(rho,sigma);
    q = QSIG_GPU(rho, sigma);
    n = ZETASIG_GPU(rho,sigma);
    F = (Z - 3.0*q)/(rho * rho);
}

__device__ inline void ENST_GPU(const double &rho, const double &sigma, double &F1, double &F2){
    F1 = 0.0;
    F2 = 0.0;

};

__device__ inline void ENSTF_GPU(const double &rho, const double &sigma, double &F1){

    F1 = 0.0;
};

__device__ inline double ENST_GPU(	const double &sigma){
    return 0.0;
};

__device__ inline double ENSTF_GPU(	const double &sigma){
    return 0.0;
};
#endif //PAWAN_GAUSSIAN_CUH
