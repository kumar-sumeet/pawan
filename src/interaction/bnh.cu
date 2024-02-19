/*! @file */

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>

#include "interaction/bnh.h"
#include "interaction/bnh_node.cuh"
#include "interaction/bnh_utility.cuh"
#include "interaction/gsl_memory.cuh"
#include "interaction/interaction_utils.h"

namespace thrust
{
    template <>
    struct minimum<double3>
    {
        __thrust_exec_check_disable__ __host__ __device__ double3 operator()(double3 const& value_1, double3 const& value_2) const
        {
            return element_min(value_1, value_2);
        }
    };
    template <>
    struct maximum<double3>
    {
        __thrust_exec_check_disable__ __host__ __device__ double3 operator()(double3 const& value_1, double3 const& value_2) const
        {
            return element_max(value_1, value_2);
        }
    };
}

namespace
{
    using namespace pawan;

    /*!
     *  Test GPU availability.
     */
    __global__ void check_gpu()
    { }

    __global__ void element_min_max(double3 const* const values, std::size_t const value_count, double3* const min_value, double3* const max_value)
    {
        __shared__ double3 cached_min_values[BNH_BLOCK_THREAD_COUNT];
        __shared__ double3 cached_max_values[BNH_BLOCK_THREAD_COUNT];

        cached_min_values[threadIdx.x] = min_value[blockIdx.x];
        cached_max_values[threadIdx.x] = max_value[blockIdx.x];

        for (auto value_index = blockIdx.x * BNH_BLOCK_THREAD_COUNT + threadIdx.x; value_index < value_count; value_index += BNH_BLOCK_COUNT * BNH_BLOCK_THREAD_COUNT)
        {
            cached_min_values[threadIdx.x] = element_min(cached_min_values[threadIdx.x], values[value_index]);
            cached_max_values[threadIdx.x] = element_max(cached_max_values[threadIdx.x], values[value_index]);
        }

        __syncthreads();

        for (auto other_thread_index = BNH_BLOCK_THREAD_COUNT / 2; other_thread_index > 0; other_thread_index /= 2)
        {
            if (threadIdx.x < other_thread_index)
            {
                cached_min_values[threadIdx.x] = element_min(cached_min_values[threadIdx.x], cached_min_values[threadIdx.x + other_thread_index]);
                cached_max_values[threadIdx.x] = element_max(cached_max_values[threadIdx.x], cached_max_values[threadIdx.x + other_thread_index]);
            }

            __syncthreads();
        }

        if (threadIdx.x > 0)
            return;

        min_value[blockIdx.x] = cached_min_values[0];
        max_value[blockIdx.x] = cached_max_values[0];
    }

    __device__ int bit_compare_3(__bnh_particle_info const* const particle_infos, std::size_t const particle_count, std::size_t const particle_index_1, std::size_t const particle_index_2)
    {
        if (particle_index_1 >= particle_count || particle_index_2 >= particle_count)
            return -1;

        auto prefix_indicator = particle_infos[particle_index_1]._code ^ particle_infos[particle_index_2]._code;
        auto prefix = sizeof(std::size_t) * CHAR_BIT - sizeof(std::size_t) * CHAR_BIT % 3;
        for (; prefix_indicator > 0; prefix_indicator >>= 1)
            --prefix;

        return prefix;
    }

    /*!
     *  Obtain Morton codes for the specified particles.
     */
    __global__ void inspect_particles(double3 const bounding_box_position, double3 const bounding_box_size,
        __bnh_particle const* const particles, std::size_t const particle_count,
        __bnh_particle_info* const particle_infos)
    {
        static constexpr std::size_t UINT21_MAX = (1 << 21) - 1;

        for (auto particle_index = blockIdx.x * BNH_BLOCK_THREAD_COUNT + threadIdx.x; particle_index < particle_count; particle_index += BNH_BLOCK_COUNT * BNH_BLOCK_THREAD_COUNT)
        {
            auto const relative_particle_position = (particles[particle_index]._position - bounding_box_position) / bounding_box_size * UINT21_MAX;

            auto& particle_info = particle_infos[particle_index];
            particle_info._index = particle_index;
            particle_info._code =
                bit_expand_3(relative_particle_position.x) << 2 |
                bit_expand_3(relative_particle_position.y) << 1 |
                bit_expand_3(relative_particle_position.z);
        }
    }

    /*!
     *  Create a hierarchy of octree nodes that structure the specified particles.
     */
    __global__ void inspect_nodes(
        __bnh_particle_info const* const particle_infos, std::size_t const particle_count,
        __bnh_node* const nodes,
        __bnh_node_info* const node_infos,
        unsigned* const node_depth_counts)
    {
        for (auto node_index = blockIdx.x * BNH_BLOCK_THREAD_COUNT + threadIdx.x; node_index < particle_count - 1; node_index += BNH_BLOCK_COUNT * BNH_BLOCK_THREAD_COUNT)
        {
            auto const delta_direction = node_index > 0
                ? (bit_compare_3(particle_infos, particle_count, node_index, node_index + 1) < bit_compare_3(particle_infos, particle_count, node_index, node_index - 1) ? -1 : 1)
                : 1;
            auto const delta_min = node_index > 0
                ? bit_compare_3(particle_infos, particle_count, node_index, node_index - delta_direction)
                : -1;

            auto node_size_cap = 2u;
            while (bit_compare_3(particle_infos, particle_count, node_index, node_index + node_size_cap * delta_direction) > delta_min)
                node_size_cap *= 2;

            auto node_size = 0u;
            for (auto t = node_size_cap / 2; t > 0; t /= 2)
            {
                if (bit_compare_3(particle_infos, particle_count, node_index, node_index + (node_size + t) * delta_direction) <= delta_min)
                    continue;

                node_size += t;
            }

            auto const particle_index_begin = node_index;
            auto const particle_index_end = particle_index_begin + node_size * delta_direction;

            auto& node_info = node_infos[node_index];
            node_info._index = node_index;
            node_info._target_depth = bit_compare_3(particle_infos, particle_count, particle_index_begin, particle_index_end);

            atomicAdd(&node_depth_counts[node_info._target_depth], 1);

            auto s = 0u;
            for (auto t = static_cast<unsigned>(std::ceil(node_size / 2.));; t = static_cast<unsigned>(std::ceil(t / 2.)))
            {
                if (bit_compare_3(particle_infos, particle_count, particle_index_begin, particle_index_begin + (s + t) * delta_direction) > node_info._target_depth)
                    s += t;

                if (t == 1)
                    break;
            }

            auto const next_index_1 = particle_index_begin + s * delta_direction - (delta_direction > 0 ? 0 : 1);
            auto const next_index_2 = next_index_1 + 1;

            auto& node = nodes[node_index];
            node._next_external_1 = (delta_direction > 0 ? particle_index_begin : particle_index_end) == next_index_1;
            node._next_index_1 = node._next_external_1 ? particle_infos[next_index_1]._index : next_index_1;
            node._next_external_2 = (delta_direction > 0 ? particle_index_end : particle_index_begin) == next_index_2;
            node._next_index_2 = node._next_external_2 ? particle_infos[next_index_2]._index : next_index_2;

            if (!node._next_external_1)
                node_infos[next_index_1]._origin_depth = node_info._target_depth;
            if (!node._next_external_2)
                node_infos[next_index_2]._origin_depth = node_info._target_depth;
        }
    }
    /*!
     *  Create pseudo-particles that summarize particle data for approximation.
     */
    __global__ void fill_nodes(double const bounding_box_diameter,
        __bnh_particle const* const particles, std::size_t const particle_count,
        __bnh_node* const nodes,
        __bnh_node_info const* const node_infos, std::size_t const node_index_begin, std::size_t const node_count)
    {
        for (auto node_index = blockIdx.x * BNH_BLOCK_THREAD_COUNT + threadIdx.x; node_index < node_count; node_index += BNH_BLOCK_COUNT * BNH_BLOCK_THREAD_COUNT)
        {
            auto const& node_info = node_infos[node_index + node_index_begin];

            auto& node = nodes[node_info._index];
            if (node._next_external_1 && node._next_external_2)
            {
                node._pseudo_particle += particles[node._next_index_1];
                node._pseudo_particle += particles[node._next_index_2];
                node._pseudo_factor = 2;
            }
            else if (node._next_external_1 && !node._next_external_2)
            {
                auto& next_node = nodes[node._next_index_2];

                node._pseudo_particle += particles[node._next_index_1];
                node._pseudo_particle += next_node._pseudo_particle;
                node._pseudo_factor = next_node._pseudo_factor + 1;

                if (next_node._diameter > 0)
                    next_node._pseudo_particle /= next_node._pseudo_factor;
            }
            else if (!node._next_external_1 && node._next_external_2)
            {
                auto& next_node = nodes[node._next_index_1];

                node._pseudo_particle += next_node._pseudo_particle;
                node._pseudo_particle += particles[node._next_index_2];
                node._pseudo_factor = next_node._pseudo_factor + 1;

                if (next_node._diameter > 0)
                    next_node._pseudo_particle /= next_node._pseudo_factor;
            }
            else
            {
                auto& next_node_1 = nodes[node._next_index_1];
                auto& next_node_2 = nodes[node._next_index_2];

                node._pseudo_particle += next_node_1._pseudo_particle;
                node._pseudo_particle += next_node_2._pseudo_particle;
                node._pseudo_factor = next_node_1._pseudo_factor + next_node_2._pseudo_factor;

                if (next_node_1._diameter > 0)
                    next_node_1._pseudo_particle /= next_node_1._pseudo_factor;
                if (next_node_2._diameter > 0)
                    next_node_2._pseudo_particle /= next_node_2._pseudo_factor;
            }

            auto const node_level = node_info._target_depth / 3;
            if (node_level == node_info._origin_depth / 3)
                continue;

            node._diameter = bounding_box_diameter / (1 << node_level);
        }
    }

    __host__ thrust::device_vector<__bnh_particle> get_particles(__wake const& wake)
    {
        std::vector<__bnh_particle> particles;
        particles.reserve(wake._numParticles);
        for (auto particle_index = 0u; particle_index < wake._numParticles; ++particle_index)
            particles.emplace_back(wake, particle_index);

        return thrust::device_vector<__bnh_particle>(particles);
    }

    __host__ std::pair<double3, double3> get_bounding_box(__wake const& wake)
    {
        thrust::device_vector<double3> particle_positions;
        particle_positions.reserve(wake._numParticles);
        for (auto particle_index = 0u; particle_index < wake._numParticles; ++particle_index)
        {
            particle_positions.push_back(
                double3{
                    .x = gsl_matrix_get(wake._position, particle_index, 0),
                    .y = gsl_matrix_get(wake._position, particle_index, 1),
                    .z = gsl_matrix_get(wake._position, particle_index, 2)});
        }

        static constexpr double3 MIN_VALUE{
            .x = -std::numeric_limits<double>::infinity(),
            .y = -std::numeric_limits<double>::infinity(),
            .z = -std::numeric_limits<double>::infinity()};
        static constexpr double3 MAX_VALUE{
            .x = std::numeric_limits<double>::infinity(),
            .y = std::numeric_limits<double>::infinity(),
            .z = std::numeric_limits<double>::infinity()};

        thrust::device_vector<double3> particle_min_positions(BNH_BLOCK_COUNT, MAX_VALUE);
        thrust::device_vector<double3> particle_max_positions(BNH_BLOCK_COUNT, MIN_VALUE);
        element_min_max<<<BNH_BLOCK_COUNT, BNH_BLOCK_THREAD_COUNT>>>(
            thrust::raw_pointer_cast(particle_positions.data()), particle_positions.size(),
            thrust::raw_pointer_cast(particle_min_positions.data()),
            thrust::raw_pointer_cast(particle_max_positions.data()));

        std::pair<double3, double3> bounding_box{
            thrust::reduce(thrust::device, particle_min_positions.begin(), particle_min_positions.end(), MAX_VALUE, thrust::minimum<double3>{}),
            thrust::reduce(thrust::device, particle_max_positions.begin(), particle_max_positions.end(), MIN_VALUE, thrust::maximum<double3>{})};
        bounding_box.second = element_next(bounding_box.second - bounding_box.first);

        return bounding_box;
    }

    __host__ void single_interact(double const nu, __wake* const source_wake, std::size_t const source_particle_index, __bnh_particle const& target_particle)
    {
        auto const& r_src = gsl_matrix_const_row(source_wake->_position, source_particle_index);
        auto const& a_src = gsl_matrix_const_row(source_wake->_vorticity, source_particle_index);
        auto const s_src = gsl_vector_get(source_wake->_radius, source_particle_index);
        auto const v_src = gsl_vector_get(source_wake->_volume, source_particle_index);

        auto const& r_trg = gsl_make_unique_vector_3(target_particle._position);
        auto const& a_trg = gsl_make_unique_vector_3(target_particle._vorticity);
        auto const s_trg = target_particle._radius;
        auto const v_trg = target_particle._volume;

        auto const& dr_trg = gsl_make_unique_vector_3(target_particle._velocity);
        auto vx = 0.0;
        auto vy = 0.0;
        auto vz = 0.0;
        auto const& da_trg = gsl_make_unique_vector_3(target_particle._retvorcity);
        auto qx = 0.0;
        auto qy = 0.0;
        auto qz = 0.0;
        INTERACT(nu, s_src, s_trg, &r_src.vector, r_trg.get(), &a_src.vector, a_trg.get(), v_src, v_trg, dr_trg.get(), da_trg.get(), vx, vy, vz, qx, qy, qz);

        auto dr_src = gsl_matrix_row(source_wake->_velocity, source_particle_index);
        gsl_vector_set(&dr_src.vector, 0, vx + gsl_vector_get(&dr_src.vector, 0));
        gsl_vector_set(&dr_src.vector, 1, vy + gsl_vector_get(&dr_src.vector, 1));
        gsl_vector_set(&dr_src.vector, 2, vz + gsl_vector_get(&dr_src.vector, 2));
        auto da_src = gsl_matrix_row(source_wake->_retvorcity, source_particle_index);
        gsl_vector_set(&da_src.vector, 0, qx + gsl_vector_get(&da_src.vector, 0));
        gsl_vector_set(&da_src.vector, 1, qy + gsl_vector_get(&da_src.vector, 1));
        gsl_vector_set(&da_src.vector, 2, qz + gsl_vector_get(&da_src.vector, 2));
    }
}

namespace pawan
{
    __bnh::__bnh(__wake* const wake, double const theta) :
        __interaction(wake),
        _theta{theta}
    {
        int gpu_count;
        if (cudaGetDeviceCount(&gpu_count) != cudaSuccess)
            throw std::runtime_error("CUDA runtime unavailable");

        check_gpu<<<1, 1>>>();
    }

    __bnh::~__bnh() = default;

    void __bnh::tree_initialize(__wake const& wake)
    {
        _nodes.clear();

        if (wake._numParticles == 0)
            return;

        auto const& wake_particles = get_particles(wake);

        auto const& wake_box = get_bounding_box(wake);
        auto const& wake_box_position = wake_box.first;
        auto const& wake_box_size = wake_box.second;

        thrust::device_vector<__bnh_particle_info> wake_particle_infos(wake_particles.size());
        inspect_particles<<<BNH_BLOCK_COUNT, BNH_BLOCK_THREAD_COUNT>>>(wake_box_position, wake_box_size,
            thrust::raw_pointer_cast(wake_particles.data()), wake_particles.size(),
            thrust::raw_pointer_cast(wake_particle_infos.data()));
        thrust::sort(thrust::device, wake_particle_infos.begin(), wake_particle_infos.end());

        thrust::device_vector<__bnh_node> wake_nodes(wake_particles.size() - 1);
        thrust::device_vector<__bnh_node_info> wake_node_infos(wake_nodes.size());
        thrust::device_vector<unsigned> wake_node_depth_counts(sizeof(std::size_t) * CHAR_BIT);
        inspect_nodes<<<BNH_BLOCK_COUNT, BNH_BLOCK_THREAD_COUNT>>>(
            thrust::raw_pointer_cast(wake_particle_infos.data()), wake_particle_infos.size(),
            thrust::raw_pointer_cast(wake_nodes.data()),
            thrust::raw_pointer_cast(wake_node_infos.data()),
            thrust::raw_pointer_cast(wake_node_depth_counts.data()));
        thrust::sort(thrust::device, wake_node_infos.begin(), wake_node_infos.end(), thrust::greater<__bnh_node_info>{});

        auto node_index_begin = 0u;
        for (auto const node_depth_count : thrust::host_vector<unsigned>(wake_node_depth_counts))
        {
            if (node_depth_count == 0)
                break;

            fill_nodes<<<BNH_BLOCK_COUNT, BNH_BLOCK_THREAD_COUNT>>>(
                l2_norm(wake_box_size),
                thrust::raw_pointer_cast(wake_particles.data()), wake_particles.size(),
                thrust::raw_pointer_cast(wake_nodes.data()),
                thrust::raw_pointer_cast(wake_node_infos.data()),
                node_index_begin,
                node_depth_count);

            node_index_begin += node_depth_count;
        }

        thrust::copy(wake_nodes.begin(), wake_nodes.end(), std::back_inserter(_nodes));
    }

    void __bnh::tree_interact(__wake* const wake) const
    {
        for (auto particle_index = 0u; particle_index < wake->_numParticles; ++particle_index)
            tree_interact(wake, particle_index, 0);
    }
    void __bnh::tree_interact(__wake* const wake, std::size_t const particle_index, std::size_t const node_index) const
    {
        auto const& node = _nodes[node_index];
        if (node._diameter > 0)
        {
            auto const& particle_position = gsl_matrix_const_row(wake->_position, particle_index);

            auto const& distance = gsl_make_unique_vector_3(node._pseudo_particle._position);
            gsl_vector_sub(distance.get(), &particle_position.vector);
            if (gsl_blas_dnrm2(distance.get()) > node._diameter / _theta)
            {
                single_interact(_nu, wake, particle_index, node._pseudo_particle);

                return;
            }
        }

        if (node._next_external_1)
        {
            if (particle_index != node._next_index_1)
                single_interact(_nu, wake, particle_index, __bnh_particle(*wake, node._next_index_1));
        }
        else
        {
            tree_interact(wake, particle_index, node._next_index_1);
        }
        if (node._next_external_2)
        {
            if (particle_index != node._next_index_2)
                single_interact(_nu, wake, particle_index, __bnh_particle(*wake, node._next_index_2));
        }
        else
        {
            tree_interact(wake, particle_index, node._next_index_2);
        }
    }
}
