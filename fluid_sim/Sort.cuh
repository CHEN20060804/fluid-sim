//cuda_common.cuh 文件
#ifndef CUDA_COMMON_H
#define CUDA_COMMON_H
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
struct Entry {
	int index;
	int cellKey;
	__host__ __device__ Entry() : index(0), cellKey(0) {}  // 默认构造
	__host__ __device__ Entry(int idx, int key) : index(idx), cellKey(key) {}
};


__global__ void SortPairs(Entry* values, int numValues, int groundHeight, int groupWidth, int stepIndex);
void launchSortPairs(Entry* values, int numValues, int groupHeight, int groupWidth, int stepIndex, int threadsPerBlock);

#endif