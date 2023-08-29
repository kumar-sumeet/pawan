#ifndef PAWAN_INTERACTION_UTILS_GPU_CUH
#define PAWAN_INTERACTION_UTILS_GPU_CUH

//TODO: this uses the highorder and not gaussian method ?!
//-> add switching mechanism??
__device__ inline void INTERACT_GPU(double nu,
                                    const double4 &source_pos,
                                    const double4 &target_pos,
                                    const double4 &source_vorticity,
                                    const double4 &target_vorticity,
                                    double3 &velocity,
                                    double3 &retvorticity);

__device__ inline void KERNEL_GPU(double rho, double sigma, double &q, double &f, double &z);
__device__ inline double ZETASIG_GPU(const double &rho, const double &sigma);
__device__ inline double QSIG_GPU(const double &rho, const double &sigma);


__device__ inline void VELOCITY_GPU(double kernel, const double4 &vorticity, const double3 &displacement, double3 &velocity);
__device__ inline void VORSTRETCH_GPU(const double &q, const double &F, const double4 &source_vorticity,
                                      const double4 &target_vorticity, double3 &displacement, double3 &retvorcity );

__device__ inline void DIFFUSION_GPU(double nu, double sigma, double Z, const double4 &source_vorticity,
                                     const double4 &target_vorticity, double3 &retvorcity );
//Math functions
template<typename A,typename B>
__device__  inline void add(A &a, B b);
__device__  inline void subtract(double3 &a, double3 b);
__device__ inline double dot_product(const double3 &a, const double3 &b);
__device__ inline double dnrm2(double3 v);
template<typename A,typename B> __device__ inline void cross(const A &a, const B &b, double3 &target);
__device__ inline void scale(double a, double3 &v);


//definitions included in header to enable inlining

/*
 * Interact version used on the GPU
 * pos x,y,z : position
 * pos.w : smoothing radius
 * vorticity x,y,z : vorticity
 * vorticity.w : volume
 */
__device__ inline void INTERACT_GPU(const double nu,
                                    const double4 &source_pos,
                                    const double4 &target_pos,
                                    const double4 &source_vorticity,
                                    const double4 &target_vorticity,
                                    double3 &velocity,
                                    double3 &retvorticity){

    double3 displacement {target_pos.x-source_pos.x,
                          target_pos.y-source_pos.y,
                          target_pos.z-source_pos.z};

    double rho = dnrm2(displacement);
    double q = 0.0, F = 0.0, Z = 0.0;
    double sigma = sqrt(0.5*(source_pos.w * source_pos.w + target_pos.w * target_pos.w));


    KERNEL_GPU(rho,sigma,q,F,Z);
    // Velocity computation of source
    double3 vel;
    VELOCITY_GPU(-q,target_vorticity,displacement,vel);
    add(velocity, vel);

    // Rate of change of vorticity computation
    VORSTRETCH_GPU(q,F,source_vorticity,target_vorticity,displacement,retvorticity);
    DIFFUSION_GPU(nu,sigma,Z,source_vorticity,target_vorticity,retvorticity);

}

__device__ inline void KERNEL_GPU(const double rho, const double sigma, double &q, double &F, double &Z) {
    Z = ZETASIG_GPU(rho,sigma);
    q = QSIG_GPU(rho, sigma);
    F = (Z - 3.0*q)/(rho * rho);

}


__device__ double QSIG_GPU(const double &rho, const double &sigma) {
    double rho_bar = rho/sigma;
    double rho_bar2 = rho_bar * rho_bar;
    double rho3 = rho*rho*rho;
    return 0.25*M_1_PI*rho_bar2*rho_bar*(rho_bar2 + 2.5)/pow(rho_bar2 + 1.0,2.5)/rho3;
}


__device__ inline double ZETASIG_GPU(	const double &rho, const double &sigma) {
    double rho_bar = rho/sigma;
    return 1.875*M_1_PI/pow(rho_bar*rho_bar + 1.0,3.5)/pow(sigma,3);
}

__device__ inline void VELOCITY_GPU(const double kernel, const double4 &vorticity,
                                    const double3 &displacement, double3 &velocity) {
    cross(vorticity,displacement,velocity);
    scale(kernel,velocity);
}

__device__ inline void VORSTRETCH_GPU(const double &q,
                                      const double &F,
                                      const double4 &source_vorticity,
                                      const double4 &target_vorticity,
                                      double3 &displacement,
                                      double3 &retvorcity ){
    // a_target x a_source
    double3 trgXsrc;
    cross(target_vorticity,source_vorticity,trgXsrc);

    // da/dt = q*(a_trg x a_src)
    double3 crossed = trgXsrc;
    scale(q,crossed);


    // da/dt = F*[disp.(a_trg x a_src)]disp
    double roaxa = dot_product(displacement,trgXsrc);

    scale(F*roaxa,displacement); //We don't need displacement after this function, so we can destroy it

    // Difference to CPU version:
    // - directly change the result vector
    // - because we only consider the source, we have to subtract instead of add
    subtract(retvorcity,crossed);
    subtract(retvorcity,displacement);

}

__device__ inline void DIFFUSION_GPU(const double nu,
                                     const double sigma,
                                     const double Z,
                                     const double4 &source_vorticity,
                                     const double4 &target_vorticity,
                                     double3 &retvorcity ){

    //.w contains volume
    // va12 = volume_target*vorticity_source
    double3 va12 = {source_vorticity.x,source_vorticity.y,source_vorticity.z};
    scale(target_vorticity.w,va12);

    // va21 = volume_source*vorticity_target
    double3 va21 = {target_vorticity.x,target_vorticity.y,target_vorticity.z};
    scale(source_vorticity.w,va21);

    // dva = 2*nu*Z*(va12 - va21)/sigma^2
    double sig12 = 0.5*sigma*sigma;
    subtract(va12,va21);
    scale(Z*nu/sig12,va12);

    // da = da + dva
    // Difference to CPU version:
    // - directly change the result vector
    // - because we only consider the source, we have to subtract instead of add
    subtract(retvorcity,va12);

}


//Math functions

/*
 * add b to a and save the result in a
 * As a template to allow usage of double4 and double3
 * only considers x,y,z
 */
template<class A, class B>
__device__  inline void add(A &a, const B b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
}

/*
 * subtract b from a and save the result in a
 */
__device__  inline void subtract(double3 &a, const double3 b) {
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
}

/*
 * dot product
 */
__device__ inline double dot_product(const double3 &a, const double3 &b) {
    return a.x * b.x
           + a.y * b.y
           + a.z * b.z;
}

/*
 * Euclidian norm
 */
__device__ inline double dnrm2(double3 v) {
    return sqrt(
            v.x * v.x
            + v.y * v.y
            + v.z * v.z
    );
}

__device__ void scale(double a, double3 &v) {
    v.x *= a;
    v.y *= a;
    v.z *= a;
}

/*
 * cross product
 * As a template to allow usage of double4 and double3
 * only considers x,y,z
 */
template<typename A,typename B>
__device__ inline void cross(const A &a, const B &b, double3 &target) {
    target.x = a.y*b.z-a.z*b.y;
    target.y = a.z*b.x-a.x*b.z;
    target.z = a.x*b.y-a.y*b.x;
}


#endif //PAWAN_INTERACTION_UTILS_GPU_CUH
