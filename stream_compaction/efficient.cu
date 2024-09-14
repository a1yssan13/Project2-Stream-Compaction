#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        using StreamCompaction::Common::kernMapToBoolean;
        using StreamCompaction::Common::kernScatter;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        /**
         * Perform the scan on the array. 
         * @param n The number of elements in the array, which is 1 << logCeil
         * @param logCeil The log base 2 of the number of elements in the array.
         * @param data The array to scan.
         */
        __global__ void kernScan(int n, int logCeil, int *data) {
            int index = threadIdx.x + blockIdx.x * blockDim.x;
            if (index >= n) {
                return; 
            }
            // up-sweep faze of array 
            for (int d = 1; d <= logCeil; d++) {
                int offset = 1 << d; // 2^d. offset used for child. 
                if ((index + 1) % offset == 0) {
                    data[index] = data[index - offset / 2] + data[index];
                }
                __syncthreads();
            }
            // set root to zero. 
            if (index == n - 1) {
                data[n - 1] = 0;
            }
            __syncthreads();
            // down-sweep faze of array  
            // set root to zero. At each pass, a node passes its value to its left 
            // child, and sets the right child to left value + this node's value. 
            for (int d = logCeil; d >= 1; d--) {
                int offset = 1 << d;
                if ((index + 1) % offset == 0) {
                    int temp = data[index];
                    data[index] = data[index] + data[index - offset / 2];
                    data[index - offset / 2] = temp;
                }
                __syncthreads();
            }
        }

        /**
         * Pad the array with zeroes to the next power of 2. 
         * @param n The number of elements in the original array.
         * @param N The next power of 2.
         * @param data The array to pad.
         */
        __global__ void kernPad(int n, int N, int *data) {
            int index = threadIdx.x + blockIdx.x * blockDim.x;
            if (index >= N || index < n) {
                return;
            }
            data[index] = 0;
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            // TODO
            int *dev_data; 
            int blockSize = 1024;  
            int logCeil = ilog2ceil(n);
            int N = 1 << logCeil; 
            dim3 numBlocks = (N + blockSize - 1) / blockSize;  

            cudaMalloc(&dev_data, N * sizeof(int));
            // pad the array with zeroes to the next power of 2
            cudaMemcpy(dev_data, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            kernPad<<<numBlocks, blockSize>>>(n, N, dev_data);
            kernScan<<<numBlocks, blockSize>>>(N, logCeil, dev_data);

            cudaMemcpy(odata, dev_data, n * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(dev_data);
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
            
            int *dev_bools, *dev_indices, *dev_data, *dev_odata; 
            int blockSize = 1024; 
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
            
            // copy dev_indices to host
            int *indices = new int[n];
            cudaMemcpy(indices, dev_indices, n * sizeof(int), cudaMemcpyDeviceToHost);
            //grab last index for total elements 
            int total_elements = idata[n-1] != 0 ? indices[n - 1] + 1 : indices[n - 1]; //exclusive scan so add 1
            //free indices array 
            delete[] indices;
            cudaFree(dev_bools);
            cudaFree(dev_indices);
            cudaFree(dev_data);
            cudaFree(dev_odata);
            
            timer().endGpuTimer();
            return total_elements;
        }
    }
}
