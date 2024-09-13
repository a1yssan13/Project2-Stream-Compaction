#include <cstdio>
#include "cpu.h"

#include "common.h"

namespace StreamCompaction {
    namespace CPU {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        /**
         * CPU scan (prefix sum).
         * For performance analysis, this is supposed to be a simple for loop.
         * (Optional) For better understanding before starting moving to GPU, you can simulate your GPU scan in this function first.
         */
        void scan(int n, int *odata, const int *idata) {
            for (int i = 0; i < n; i++) {
                if (i == 0) {
                    odata[i] = 0;
                }
                else {
                    odata[i] = odata[i - 1] + idata[i - 1];
                }
            }
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            int count = 0; 
            for (int i = 0; i < n; i++) {
                if (idata[i] != 0) {
                    odata[count++] = idata[i];
                }
            }
            timer().endCpuTimer();
            return count;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            int *temp = new int[n];
            int* temp_idx = new int[n]; 
            // array of 1 if number exists, 0 if not
            for (int i = 0; i < n; i++) {
                temp[i] = idata[i] != 0 ? 1 : 0; 
            }
            scan(n, temp_idx, temp);
            // scatter the values back into the original array
            for (int i = 0; i < n; i++) {
                if (idata[i] != 0) {
                    odata[temp_idx[i]] = idata[i];
                }
            }
            timer().endCpuTimer();
            return temp_idx[n-1];
        }
    }
}
