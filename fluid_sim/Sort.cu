#include "Sort.cuh"
#include <cuda_runtime_api.h>

__global__ void SortPairs(Entry* values, int numValues, int groupHeight, int groupWidth, int stepIndex)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= numValues) return;

	int h = i & ((groupHeight + 1) * (1 / groupWidth));
	int indexLow = h + (groupHeight + 1) * (i / groupWidth);
	int indexHigh = indexLow + (stepIndex == 0 ? groupHeight - 2 * h : (groupHeight + 1) / 2);

	if (indexHigh >= numValues) return;
	int valueLow = values[indexLow].cellKey;
	int valueHigh = values[indexHigh].cellKey;
	if (valueLow > valueHigh)
	{
		values[indexLow].cellKey = valueHigh;
		values[indexHigh].cellKey = valueLow;
	}

}
void launchSortPairs(Entry* values, int numValues, int groupHeight, int groupWidth, int stepIndex, int threadsPerBlock)
{
	int blocksPerGrid = (numValues + threadsPerBlock - 1) / threadsPerBlock;
	SortPairs <<< blocksPerGrid, threadsPerBlock >>> (values, numValues, groupHeight, groupWidth, stepIndex);
	cudaDeviceSynchronize();
}