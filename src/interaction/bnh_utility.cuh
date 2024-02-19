#pragma once

/*! @file */

#include <cstdint>

/*!
 *  Element-wise subtract two specified vectors.
 */
__host__ __device__ double3& operator-=(double3&, double3 const&);
/*! @copybrief operator-=(double3&, double3 const&) */
__host__ __device__ double3 operator-(double3, double3 const&);

/*!
 *  Element-wise scale a specified vector.
 */
__host__ __device__ double3& operator*=(double3&, double);
/*! @copybrief operator*=(double3&, double) */
__host__ __device__ double3 operator*(double3, double);

/*!
 *  Element-wise divide two specified vectors.
 */
__host__ __device__ double3& operator/=(double3&, double3 const&);
/*! @copybrief operator/=(double3&, double3 const&) */
__host__ __device__ double3 operator/(double3, double3 const&);

/*!
 *  Obtain the euclidean (L2) norm of the specified vector.
 */
__host__ __device__ double l2_norm(double3 const&);

/*!
 *  Obtain the next representable floating-point value in positive direction for every element of the specified vector.
 */
__host__ __device__ double3 element_next(double3 const&);

/*!
 *  Obtain the element-wise minimum of two specified vectors.
 */
__host__ __device__ double3 element_min(double3 const&, double3 const&);
/*!
 *  Obtain the element-wise maximum of two specified vectors.
 */
__host__ __device__ double3 element_max(double3 const&, double3 const&);

/*!
 *  Triple a number's bit width by prepending every bit with two zeros.
 *
 *  @param value
 *      21-bit number (max @c 0x00000000001FFFFF)
 *
 *  @returns
 *      63-bit number (max @c 0x1249249249249249)
 *
 *  @remark
 *      The result's most significant bit is always zero.
 *  @remark
 *      Origin of the algorithm: <a href="https://stackoverflow.com/a/18528775">https://stackoverflow.com/a/18528775</a>
 */
__host__ __device__ std::uint64_t bit_expand_3(std::uint64_t value);
