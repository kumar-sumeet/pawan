#include "interaction/bnh.h"

namespace pawan
{
    void __bnh::interact(__wake* const wake)
    {
        tree_initialize(*wake);
        tree_interact(wake);
    }
}

static_assert(BNH_BLOCK_COUNT > 0, "Number of blocks must be greater than zero.");
static_assert((BNH_BLOCK_THREAD_COUNT & (BNH_BLOCK_THREAD_COUNT - 1)) == 0, "Number of threads per block must be a power of two and greater than zero.");
