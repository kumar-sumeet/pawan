#pragma once

/*! @file */

#include <deque>

#include "interaction/interaction.h"

#define BNH_BLOCK_COUNT 256
#define BNH_BLOCK_THREAD_COUNT 256

namespace pawan
{
    class __bnh_node;
    /*!
     *  CUDA-accelerated wake interaction using Barnes-Hut approximation
     */
    class __bnh : public __interaction
    {
        std::deque<__bnh_node> _nodes;

        double _theta;

    public:
        __bnh(__wake*, double);

        ~__bnh();

        __bnh(__bnh const&) = delete;
        __bnh& operator=(__bnh const&) = delete;

        __bnh(__bnh&&) = delete;
        __bnh& operator=(__bnh&&) = delete;

    private:
#ifndef __CUDACC__
        void interact(__wake*) override;
#endif

        /*!
         *  Create the Barnes-Hut interaction tree of a specified wake.
         *
         *  @remark
         *      The algorithm is based on the paper <a href="https://research.nvidia.com/publication/2012-06_maximizing-parallelism-construction-bvhs-octrees-and-k-d-trees">Maximizing Parallelism in the Construction of BVHs, Octrees, and k-d Trees</a>.
         */
        void tree_initialize(__wake const&);

        void tree_interact(__wake*) const;
        void tree_interact(__wake*, std::size_t, std::size_t) const;
    };
}
