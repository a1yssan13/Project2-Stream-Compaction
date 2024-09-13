#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        __global__ void kernScan(int n, int *data) {
            int index = threadIdx.x + blockIdx.x * blockDim.x;
            if (index >= n) {
                return; 
            }
            // up-sweep faze of array 
            for (int d = 0; d < ilog2ceil(n); d++) {
                int offset = 1 << d; // 2^d. offset used for child. 
                if ((index + 1) / offset % 2 == 0) {
                    data[index] = data[index - offset] + data[index]; 
                __syncthreads(); 
            }
            // down-sweep faze of array  
            // set root to zero. At each pass, a node passes its value to its left 
            // child, and sets the right child to left value + this node's value. 
            for (int d = ilog2ceil(n) - 1; d >= 0; d--) {
                int offset = 1 << d;
                if ((index + 1) / offset % 2 == 0) {
                    int temp = data[index]; 
                    data[index] = data[index] + data[index - offset]; 
                    data[index - offset] = temp; 
                }
                __syncthreads(); 
            }
        }
        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
            // TODO
            int *dev_data; 
            dim3 blockSize = 1024;  
            dim3 numBlocks = (n + blockSize - 1) / blockSize;  

            cudaMalloc(&dev_data, n * sizeof(int));
            cudaMemcpy(dev_data, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            kernScan<<<numBlocks, blockSize>>>(n, dev_data);

            cudaMemcpy(odata, dev_data, n * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(dev_data);

            timer().endGpuTimer();
        }

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
            
            int *dev_bools, *dev_indices; 
            dim3 blockSize = 1024; 
            dim3 numBlocks = (n + blockSize - 1) / blockSize; 

            scan(n, odata, idata);

            cudaMalloc(&dev_bools, n * sizeof(int));
            cudaMalloc(&dev_indices, n * sizeof(int));
            cudaMalloc(&dev_data, n * sizeof(int));
            cudaMalloc(&dev_odata, n * sizeof(int));
            cudaMemcpy(dev_data, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            
            kernMapToBoolean<<<numBlocks, blockSize>>>(n, dev_bools, dev_data);
            kernScatter<<<numBlocks, blockSize>>>(n, dev_odata, dev_data, dev_bools, dev_indices);

            cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(dev_bools);
            cudaFree(dev_indices);
            cudaFree(dev_data);
            cudaFree(dev_odata);
            
            timer().endGpuTimer();
            return -1;
        }
    }
}
