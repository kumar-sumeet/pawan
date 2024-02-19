#include "interaction/bnh_particle.cuh"

namespace
{
    __device__ void add_3(double3* const value_1, double3 const& value_2)
    {
        value_1->x += value_2.x;
        value_1->y += value_2.y;
        value_1->z += value_2.z;
    }
    __device__ void div_3(double3* const value_1, double const value_2)
    {
        value_1->x /= value_2;
        value_1->y /= value_2;
        value_1->z /= value_2;
    }

    __host__ double3 gsl_matrix_get_row_3(gsl_matrix const& matrix, std::size_t const index)
    {
        return {
            .x = gsl_matrix_get(&matrix, index, 0),
            .y = gsl_matrix_get(&matrix, index, 1),
            .z = gsl_matrix_get(&matrix, index, 2)};
    }
}

namespace pawan
{
    __device__ __bnh_particle::__bnh_particle() :
        _radius{0},
        _volume{0},
        _birthstrength{0},
        _position{0},
        _velocity{0},
        _vorticity{0},
        _vorticityfield{0},
        _retvorcity{0}
    { }

    __host__ __bnh_particle::__bnh_particle(__wake const& wake, std::size_t const index) :
        _radius{gsl_vector_get(wake._radius, index)},
        _volume{gsl_vector_get(wake._volume, index)},
        _birthstrength{gsl_vector_get(wake._birthstrength, index)},
        _position{gsl_matrix_get_row_3(*wake._position, index)},
        _velocity{gsl_matrix_get_row_3(*wake._velocity, index)},
        _vorticity{gsl_matrix_get_row_3(*wake._vorticity, index)},
        _vorticityfield{gsl_matrix_get_row_3(*wake._vorticityfield, index)},
        _retvorcity{gsl_matrix_get_row_3(*wake._retvorcity, index)}
    { }

    __device__ __bnh_particle& __bnh_particle::operator+=(__bnh_particle const& other)
    {
        _radius += other._radius;
        _volume += other._volume;
        _birthstrength += other._birthstrength;

        add_3(&_position, other._position);
        add_3(&_velocity, other._velocity);
        add_3(&_vorticity, other._vorticity);
        add_3(&_vorticityfield, other._vorticityfield);
        add_3(&_retvorcity, other._retvorcity);

        return *this;
    }
    __device__ __bnh_particle& __bnh_particle::operator/=(double const value)
    {
        _radius /= value;
        _volume /= value;
        _birthstrength /= value;

        div_3(&_position, value);
        div_3(&_velocity, value);
        div_3(&_vorticity, value);
        div_3(&_vorticityfield, value);
        div_3(&_retvorcity, value);

        return *this;
    }

    __device__ __bnh_particle_info::__bnh_particle_info() :
        _index{0},
        _code{0}
    { }

    __device__ bool __bnh_particle_info::operator<(__bnh_particle_info const& other) const
    {
        return _code < other._code;
    }
}
