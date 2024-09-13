#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

#define blockSize 256

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        // TODO: __global__
        __global__ void naiveScanKernel(int n, int d, int *odata, const int *idata) {
            int index = threadIdx.x + blockIdx.x * blockDim.x;
            if (index < n) {
                if (index >= pow(2, d)) {
                    odata[index] = idata[index - pow(2, d-1)] + idata[index];
                } else {
                    odata[index] = idata[index];
                }
            }
        }
        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
            int *dev_idata;
            int *dev_odata;

            dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize); 

            cudaMalloc((void**)&dev_idata, n * sizeof(int));
            cudaMalloc((void**)&dev_odata, n * sizeof(int));
            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            
            for (int d = 1; d <= ilog2ceil(n); d++) {
                naiveScanKernel<<<fullBlocksPerGrid, blockSize>>>(n, d, dev_odata, dev_idata);
                swap(dev_idata, dev_odata);
            }
            
            cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(dev_idata);
            cudaFree(dev_odata);

            timer().endGpuTimer();
        }
    }
}
