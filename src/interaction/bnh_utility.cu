#include "interaction/bnh_utility.cuh"

__host__ __device__ double3& operator-=(double3& value_1, double3 const& value_2)
{
    value_1.x -= value_2.x;
    value_1.y -= value_2.y;
    value_1.z -= value_2.z;

    return value_1;
}
__host__ __device__ double3 operator-(double3 value_1, double3 const& value_2)
{
    return value_1 -= value_2;
}

__host__ __device__ double3& operator*=(double3& value_1, double const value_2)
{
    value_1.x *= value_2;
    value_1.y *= value_2;
    value_1.z *= value_2;

    return value_1;
}
__host__ __device__ double3 operator*(double3 value_1, double const value_2)
{
    return value_1 *= value_2;
}

__host__ __device__ double3& operator/=(double3& value_1, double3 const& value_2)
{
    value_1.x /= value_2.x;
    value_1.y /= value_2.y;
    value_1.z /= value_2.z;

    return value_1;
}
__host__ __device__ double3 operator/(double3 value_1, double3 const& value_2)
{
    return value_1 /= value_2;
}

__host__ __device__ double l2_norm(double3 const& value)
{
    return std::sqrt(
        value.x * value.x +
        value.y * value.y +
        value.z * value.z);
}

__host__ __device__ double3 element_next(double3 const& value)
{
    return {
        .x = std::nextafter(value.x, value.x + 1),
        .y = std::nextafter(value.y, value.y + 1),
        .z = std::nextafter(value.z, value.z + 1)};
}

__host__ __device__ double3 element_min(double3 const& value_1, double3 const& value_2)
{
    return {
        .x = value_1.x < value_2.x ? value_1.x : value_2.x,
        .y = value_1.y < value_2.y ? value_1.y : value_2.y,
        .z = value_1.z < value_2.z ? value_1.z : value_2.z};
}
__host__ __device__ double3 element_max(double3 const& value_1, double3 const& value_2)
{
    return {
        .x = value_1.x > value_2.x ? value_1.x : value_2.x,
        .y = value_1.y > value_2.y ? value_1.y : value_2.y,
        .z = value_1.z > value_2.z ? value_1.z : value_2.z};
}

__host__ __device__ std::uint64_t bit_expand_3(std::uint64_t value)
{
    value &= 0x00'00'00'00'00'1F'FF'FF;

    value |= value << 32;
    value &= 0x00'1F'00'00'00'00'FF'FF;

    value |= value << 16;
    value &= 0x00'1F'00'00'FF'00'00'FF;

    value |= value << 8;
    value &= 0x10'0F'00'F0'0F'00'F0'0F;

    value |= value << 4;
    value &= 0x10'C3'0C'30'C3'0C'30'C3;

    value |= value << 2;
    value &= 0x12'49'24'92'49'24'92'49;

    return value;
}
