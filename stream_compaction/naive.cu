#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

#define blockSize 256

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        using StreamCompaction::Common::inclusive_to_exclusive_scan; 

        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        __global__ void kernNaiveScan(int n, int d, int *odata, const int *idata) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index < n) {
                if (index >= pow(2, d-1)) {
                    odata[index] = idata[index - (int) pow(2, d-1)] + idata[index];
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
                kernNaiveScan <<<fullBlocksPerGrid, blockSize>>>(n, d, dev_odata, dev_idata);
                std::swap(dev_idata, dev_odata);
            }
            cudaMemcpy(odata, dev_idata, n * sizeof(int), cudaMemcpyDeviceToHost);
            inclusive_to_exclusive_scan(odata, n); 

            cudaFree(dev_idata);
            cudaFree(dev_odata);

            timer().endGpuTimer();
        }
    }
}
