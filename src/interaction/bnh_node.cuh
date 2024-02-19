#pragma once

/*! @file */

#include "interaction/bnh_particle.cuh"

namespace pawan
{
    struct __bnh_node
    {
        __bnh_particle _pseudo_particle;
        std::size_t _pseudo_factor;

        double _diameter;

        bool _next_external_1;
        std::size_t _next_index_1;

        bool _next_external_2;
        std::size_t _next_index_2;

        __device__ explicit __bnh_node();
    };
    struct __bnh_node_info
    {
        std::size_t _index;

        unsigned _origin_depth;
        unsigned _target_depth;

        __device__ explicit __bnh_node_info();

        __device__ bool operator>(__bnh_node_info const&) const;
    };
}
