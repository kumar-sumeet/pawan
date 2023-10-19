#ifndef PAWAN_INTERACTION_UTILS_GPU_CUH
#define PAWAN_INTERACTION_UTILS_GPU_CUH

#define HIGHORDER_GPU 1
#define GAUSSIAN_GPU 0

#if HIGHORDER_GPU
#include "src/interaction/highorder.cuh"
#elif GAUSSIAN_GPU
#include "src/interaction/gaussian.cuh"
#endif

__device__ inline void INTERACT_GPU(double nu,
                                    const double4 &source_pos,
                                    const double4 &target_pos,
                                    const double4 &source_vorticity,
                                    const double4 &target_vorticity,
                                    double3 &velocity,
                                    double3 &retvorticity);

__device__ inline void VELOCITY_GPU(double kernel, const double4 &vorticity, const double3 &displacement, double3 &velocity);
__device__ inline void VORSTRETCH_GPU(const double &q, const double &F, const double4 &source_vorticity,
                                      const double4 &target_vorticity, double3 &displacement, double3 &retvorcity );

__device__ inline void DIFFUSION_GPU(double nu, double sigma, double n, const double4 &source_vorticity,
                                     const double4 &target_vorticity, double3 &retvorcity );
__device__ inline double3 getvorticity(const double4 *particles, const unsigned int i);
__device__ inline double3 getlinearimpulse(const double4 *particles, const unsigned int i);
__device__ inline double3 getangularimpulse(const double4 *particles, const unsigned int i);
__device__ inline double3 getangularimpulse(const double4 *particles, const unsigned int i);
__device__ inline double3 getZc(const double4 *particles, const unsigned int i);
__device__ inline double3 getVi(const double4 *particles, const unsigned int i);
//Math functions
template<typename A,typename B>
__device__  inline void add(A &a, B b);
template<class A, class B>
__device__  inline void scaleadd(A &a, const B &b, const double &c);
__device__  inline void subtract(double3 &a, double3 b);
template<typename A,typename B>
__device__ inline double dot_product(const A &a, const B &b);
__device__ inline double dnrm2(double3 v);
__device__ inline double dnrm2(double4 v);
template<typename A,typename B>
__device__ inline void cross(const A &a, const B &b, double3 &target);
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
    double q = 0.0, F = 0.0, Z = 0.0, n = 0.0;
    double sigma = sqrt(0.5*(source_pos.w * source_pos.w + target_pos.w * target_pos.w));


    KERNEL_GPU(rho,sigma,q,F,Z,n);
    // Velocity computation of source
    double3 vel;
    VELOCITY_GPU(-q,target_vorticity,displacement,vel);
    add(velocity, vel);

    // Rate of change of vorticity computation
    //VORSTRETCH_GPU(q,F,source_vorticity,target_vorticity,displacement,retvorticity);
    DIFFUSION_GPU(nu,sigma,n,source_vorticity,target_vorticity,retvorticity);

}

__device__ inline void VELOCITY_GPU(const double kernel, const double4 &vorticity,
                                    const double3 &displacement, double3 &velocity) {
    cross(vorticity,displacement,velocity);
    scale(kernel,velocity);
}

//classical scheme
/*__device__ inline void VORSTRETCH_GPU(const double &q,
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

    // da/dt = F*[(disp.a_src)(a_trg x disp)]
    double d_dot_asrc = dot_product(displacement,source_vorticity);
    double3 trgXdisp;
    cross(target_vorticity,displacement,trgXdisp);
    scale(F*d_dot_asrc,trgXdisp);

    add(retvorcity,crossed);
    add(retvorcity,trgXdisp);
}*/
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
                                     const double n,
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
    scale(n*nu/sig12,va12);

    // da = da + dva
    // Difference to CPU version:
    // - directly change the result vector
    // - because we only consider the source, we have to subtract instead of add
    subtract(retvorcity,va12);

}

__device__ inline double3 getvorticity(const double4 *particles, const unsigned int i){
    double3 O;
    O.x = particles[2 * i + 1].x;
    O.y = particles[2 * i + 1].y;
    O.z = particles[2 * i + 1].z;

    return O;
}
__device__ inline double3 getlinearimpulse(const double4 *particles, const unsigned int i){
    double3 I;
    cross(particles[2 * i],particles[2 * i + 1],I);
    I.x = (1.0/2.0)*I.x;
    I.y = (1.0/2.0)*I.y;
    I.z = (1.0/2.0)*I.z;

    return I;
}
__device__ inline double3 getangularimpulse(const double4 *particles, const unsigned int i){
    double3 A1,A;
    cross(particles[2 * i],particles[2 * i + 1],A1);
    cross(particles[2 * i],A1,A);
    A.x = (1.0/3.0)*A.x;
    A.y = (1.0/3.0)*A.y;
    A.z = (1.0/3.0)*A.z;

    return A;
}
__device__ inline double3 getZc(const double4 *particles, const unsigned int i){
    double3 V = {particles[2 * i + 1].x, particles[2 * i + 1].y, particles[2 * i + 1].z};
    double oy = dnrm2(V);

    double3 result = {oy * particles[2 * i].z * (particles[2 * i].x * particles[2 * i].x + particles[2 * i].y * particles[2 * i].y),
                      oy * (particles[2 * i].x * particles[2 * i].x + particles[2 * i].y * particles[2 * i].y),
                      0.0};   //{numerator, denominator, Zc value could go here later}

    return result;
}
__device__ inline double3 getVi(const double4 &source_pos, const double4 &source_vor, const double3 &pos){
    double3 displacement {pos.x-source_pos.x,
                          pos.y-source_pos.y,
                          pos.z-source_pos.z};
    double rho = dnrm2(displacement);
    double sigma = source_pos.w;
    double q = QSIG_GPU(rho,sigma);
    double3 partContribVel {0,0,0};
    VELOCITY_GPU(-q,source_vor,displacement,partContribVel);
    //partContribVel.x = 1;
    //partContribVel.y = 1;
    //partContribVel.z = 1;
    return partContribVel;
}
__device__ inline void getVi(const double4 &source_pos,
                             const double4 &source_vor,
                             const double3 &pos,
                             double3 &partVel){
    double3 displacement {pos.x-source_pos.x,
                          pos.y-source_pos.y,
                          pos.z-source_pos.z};
    double rho = dnrm2(displacement);
    double sigma = source_pos.w;
    double q = QSIG_GPU(rho,sigma);
    double3 partContribVel {0,0,0};
    VELOCITY_GPU(-q,source_vor,displacement,partContribVel);
    //printf("(%f, %f, %f)--",partContribVel.x,partContribVel.y,partContribVel.z);
    add(partVel, partContribVel);
/*    partVel.x +=1;
    partVel.y +=1;
    partVel.z +=1;
 */
}

__device__ inline void DIVFREEOMEGA_GPU(const double4 &source_pos,
                                      const double4 &target_pos,
                                      const double4 &target_vorticity,
                                      double3 &divfreeomega) {
    double3 displacement {source_pos.x-target_pos.x,
                          source_pos.y-target_pos.y,
                          source_pos.z-target_pos.z};

    double rho = dnrm2(displacement);
    double rho2 = rho*rho;
    double q = 0.0, Z = 0.0;
    double sigma = sqrt(0.5*(source_pos.w * source_pos.w + target_pos.w * target_pos.w));
    Z = ZETASIG_GPU(rho,sigma);
    q = QSIG_GPU(rho, sigma);
    double factor1 = Z - q;
    double factor2 = dot_product(displacement,target_vorticity);
    factor2 *= (3*q - Z)/rho2;

    divfreeomega.x += factor1 * target_vorticity.x + factor2 * displacement.x;
    divfreeomega.y += factor1 * target_vorticity.y + factor2 * displacement.y;
    divfreeomega.z += factor1 * target_vorticity.z + factor2 * displacement.z;
}
__device__ inline void GRIDSOL_GPU(const double3 &node_pos,
                                    const double4 &part_pos,
                                    const double4 &part_vor,
                                    double3 &node_vel,
                                   double3 &node_vor){

    double3 displacement {part_pos.x-node_pos.x,
                          part_pos.y-node_pos.y,
                          part_pos.z-node_pos.z};

    double rho = dnrm2(displacement);
    double q = 0.0, F = 0.0, Z = 0.0, n = 0.0;
    double sigma = part_pos.w;

    KERNEL_GPU(rho,sigma,q,F,Z,n);
    // Velocity computation of source
    double3 vel;
    VELOCITY_GPU(-q,part_vor,displacement,vel);

    add(node_vel, vel);
    scaleadd(node_vor, part_vor, Z);
    /*node_vor.x += Z*part_vor.x;
    node_vor.y += Z*part_vor.y;
    node_vor.z += Z*part_vor.z;
*///    node_vel.x +=1;node_vel.y +=1;node_vel.z +=1;
//node_vor.x +=1;node_vor.y +=1;node_vor.z +=1;
}
__device__ inline void ENSTROPHY(const double4 &source_pos,
                                   const double4 &target_pos,
                                   const double4 &source_vorticity,
                                   const double4 &target_vorticity,
                                   double &partDiagContrib){

    double3 displacement {target_pos.x-source_pos.x,
                          target_pos.y-source_pos.y,
                          target_pos.z-source_pos.z};
    double rho = dnrm2(displacement);
    double sigma = sqrt(0.5*(source_pos.w * source_pos.w + target_pos.w * target_pos.w));
    double F1 = 0.0;
    double F2 = 0.0;
    ENST_GPU(rho,sigma,F1,F2);

    // (a1.a2)
    double a1a2 = dot_product(source_vorticity,target_vorticity);
    // (a1.x12)
    double a1x12 = dot_product(source_vorticity,displacement);
    // (a2.x12)
    double a2x12 = dot_product(target_vorticity,displacement);

    // F1.(a1.a2) + F2(a1.x12).(a2.x12)
    partDiagContrib += (F1 * a1a2 + F2 * a1x12 * a2x12);
}

__device__ inline void ENSTROPHYF(const double4 &source_pos,
                                   const double4 &target_pos,
                                   const double4 &source_vorticity,
                                   const double4 &target_vorticity,
                                    double &partDiagContrib){

    double3 displacement {target_pos.x-source_pos.x,
                          target_pos.y-source_pos.y,
                          target_pos.z-source_pos.z};
    double rho = dnrm2(displacement);
    double sigma = sqrt(0.5*(source_pos.w * source_pos.w + target_pos.w * target_pos.w));
    double F1 = 0.0;
    ENSTF_GPU(rho,sigma,F1);

    double a1a2 = dot_product(source_vorticity,target_vorticity);

    partDiagContrib += (F1 * a1a2);
}

__device__ inline void HELICITY(const double4 &source_pos,
                                    const double4 &target_pos,
                                    const double4 &source_vorticity,
                                    const double4 &target_vorticity,
                                  double &partDiagContrib){

    double3 displacement {target_pos.x-source_pos.x,
                          target_pos.y-source_pos.y,
                          target_pos.z-source_pos.z};
    double rho = dnrm2(displacement);
    double sigma = sqrt(0.5*(source_pos.w * source_pos.w + target_pos.w * target_pos.w));
    double q = QSIG_GPU(rho,sigma);

    // a2 x a1
    double3 trgXsrc;
    cross(target_vorticity,source_vorticity,trgXsrc);
    // x12.(a2 x a1)
    double roaxa = dot_product(displacement,trgXsrc);

    partDiagContrib += (q * roaxa);
}

__device__ inline void KINETICENERGY(const double4 &source_pos,
                                  const double4 &target_pos,
                                  const double4 &source_vorticity,
                                  const double4 &target_vorticity,
                                       double &partDiagContrib){

    double3 displacement {target_pos.x-source_pos.x,
                          target_pos.y-source_pos.y,
                          target_pos.z-source_pos.z};
    double rho2 = dot_product(displacement,displacement);
    double sigma2 = 0.5*(source_pos.w * source_pos.w + target_pos.w * target_pos.w);

    // a1.a2
    double a1a2 = dot_product(target_vorticity,source_vorticity);
    // x12.a1
    double x12a1 = dot_product(displacement,source_vorticity);
    // x12.a2
    double x12a2 = dot_product(displacement,target_vorticity);

    partDiagContrib += (((rho2 + 2.0 * sigma2) * a1a2 + (x12a1 * x12a2)) / pow(rho2 + sigma2, 1.5) / 16.0 / M_PI);
}

__device__ inline void KINETICENERGYF(const double4 &source_pos,
                                       const double4 &target_pos,
                                       const double4 &source_vorticity,
                                       const double4 &target_vorticity,
                                        double &partDiagContrib){

    double3 displacement {target_pos.x-source_pos.x,
                          target_pos.y-source_pos.y,
                          target_pos.z-source_pos.z};
    double rho2 = dot_product(displacement,displacement);
    double sigma2 = 0.5*(source_pos.w * source_pos.w + target_pos.w * target_pos.w);

    // a1.a2
    double a1a2 = dot_product(target_vorticity,source_vorticity);

    // (1/8 pi).(rho^2 + 1.5*sigma^2)*(a1.a2) /(rho^2 + sigma^2)^3/2
    partDiagContrib += ((rho2 + 1.5 * sigma2) * a1a2 / pow(rho2 + sigma2, 1.5) / 8.0 / M_PI);
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

template<class A, class B>
__device__  inline void scaleadd(A &a, const B &b, const double &c) {
    a.x += c*b.x;
    a.y += c*b.y;
    a.z += c*b.z;
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
template<typename A,typename B>
__device__ inline double dot_product(const A &a, const B &b) {
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
__device__ inline double dnrm2(double4 v) {
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
