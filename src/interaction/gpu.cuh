#ifndef PAWAN_GPU_CUH
#define PAWAN_GPU_CUH

#include "parallel.h"
#include "../wake/wake.h"


namespace pawan {

    class gpu : public __parallel {

        void interact() override;

    public:
        gpu(__wake *W);
        gpu(__wake *W1, __wake *W2);
    };



} // pawan

#endif //PAWAN_GPU_CUH