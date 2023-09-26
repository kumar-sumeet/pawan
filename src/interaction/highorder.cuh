//
// Created by ge56beh on 11.09.23.
//

#ifndef PAWAN_HIGHORDER_CUH
#define PAWAN_HIGHORDER_CUH

__device__ inline double ZETASIG_GPU(const double &rho, const double &sigma);
__device__ inline double QSIG_GPU(const double &rho, const double &sigma);
__device__ inline double ETASIG_GPU(const double &rho, const double &sigma);
__device__ inline void KERNEL_GPU(double rho, double sigma, double &q, double &f, double &z, double &n);
__device__ inline void ENST_GPU(const double &rho, const double &sigma, double &F1, double &F2);
__device__ inline void ENSTF_GPU(const double &rho, const double &sigma, double &F1);
__device__ inline double ENST_GPU(	const double &sigma);
__device__ inline double ENSTF_GPU(	const double &sigma);

__device__ inline double ZETASIG_GPU(	const double &rho, const double &sigma) {
    double rho_bar = rho/sigma;
    return 1.875*M_1_PI/pow(rho_bar*rho_bar + 1.0,3.5)/pow(sigma,3);
}

__device__ inline double QSIG_GPU(const double &rho, const double &sigma) {
    double rho_bar = rho/sigma;
    double rho_bar2 = rho_bar * rho_bar;
    double rho3 = rho*rho*rho;
    return 0.25*M_1_PI*rho_bar2*rho_bar*(rho_bar2 + 2.5)/pow(rho_bar2 + 1.0,2.5)/rho3;
}

__device__ inline double ETASIG_GPU(	const double &rho, const double &sigma) {
    double rho_bar = rho/sigma;
    return 13.125*M_1_PI/pow(rho_bar*rho_bar + 1.0,4.5)/pow(sigma,3);
}

__device__ inline void KERNEL_GPU(const double rho, const double sigma, double &q, double &F, double &Z, double &n) {
    Z = ZETASIG_GPU(rho,sigma);
    q = QSIG_GPU(rho, sigma);
    n = ETASIG_GPU(rho,sigma);
    F = (Z - 3.0*q)/(rho * rho);
}

__device__ inline void ENST_GPU(const double &rho, const double &sigma, double &F1, double &F2){
    double rho2 = rho * rho;
    double sig2 = sigma * sigma;
    double factor = (M_1_PI/8.0)/sqrt(pow(rho2 + sig2,7));
    F1 = factor*( 2.0*rho2*rho2 +  7.0*sig2*rho2 + 20.0*sig2*sig2);
    F2 = factor*(-6.0*rho2*rho2 - 27.0*sig2*rho2 - 21.0*sig2*sig2)/(rho2 + sig2);

};

__device__ inline void ENSTF_GPU(const double &rho, const double &sigma, double &F1){
    double rho2 = rho * rho;
    double sig2 = sigma * sigma;
    F1 = (15.0*M_1_PI*sig2*sig2/8.0)/sqrt(pow(rho2 + sig2,7));
};

__device__ inline double ENST_GPU(	const double &sigma){
    return 5.0*M_1_PI/2.0/pow(sigma,3);
};

__device__ inline double ENSTF_GPU(	const double &sigma){
    return 15.0*M_1_PI/8.0/pow(sigma,3);
};
#endif //PAWAN_HIGHORDER_CUH
