#pragma once

#include "integration.h"
#include "interaction/gpu.cuh"
#include "wake/wake.h"


namespace pawan{
class gpu_euler : public __integration{

	public:
        gpu_euler(const double &t, const size_t &n);
		
		~gpu_euler() = default;

     void integrate(__system *S,
                           __io *IO,
                           NetworkInterfaceTCP<OPawanRecvData,OPawanSendData> *networkCommunicatorTest,
                           bool diagnose=false) override;

     void integrate(__system *S,
                           __io *IO,
                           bool diagnose=false
                           ) override;
};
}
