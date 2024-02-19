#include "interaction/bnh_node.cuh"

namespace pawan
{
    __device__ __bnh_node::__bnh_node() :
        _pseudo_factor{0},
        _diameter{0},
        _next_external_1{false},
        _next_index_1{0},
        _next_external_2{false},
        _next_index_2{0}
    { }

    __device__ __bnh_node_info::__bnh_node_info() :
        _index{0},
        _origin_depth{0},
        _target_depth{0}
    { }

    __device__ bool __bnh_node_info::operator>(__bnh_node_info const& other) const
    {
        return _target_depth > other._target_depth;
    }
}
