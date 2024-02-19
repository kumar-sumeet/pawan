#pragma once

/*! @file */

#include "wake/wake.h"

namespace pawan
{
    struct __bnh_particle
    {
        double _radius;
        double _volume;
        double _birthstrength;

        double3 _position;
        double3 _velocity;
        double3 _vorticity;
        double3 _vorticityfield;
        double3 _retvorcity;

        /*!
         *  Initialize a particle with all values set to zero.
         */
        __device__ explicit __bnh_particle();

        /*!
         *  Initialize a particle by copying an existing one.
         *
         *  @param wake
         *      Particle source
         *  @param index
         *      Index of a particle inside the specified wake
         */
        __host__ explicit __bnh_particle(__wake const& wake, std::size_t index);

        __device__ __bnh_particle& operator+=(__bnh_particle const&);
        __device__ __bnh_particle& operator/=(double const);
    };
    struct __bnh_particle_info
    {
        std::size_t _index;

        std::uint64_t _code;

        __device__ explicit __bnh_particle_info();

        __device__ bool operator<(__bnh_particle_info const&) const;
    };
}
