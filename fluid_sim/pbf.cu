#pragma once
#include "pbf.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <algorithm>
#include <sm_20_atomic_functions.h>
#include <device_atomic_functions.h>
#define M_PI 3.14159265358979323846
#define MAX_NEIGHBORS 30
const int p1 = 73856093;
const int p2 = 19349663;
struct PBFSolverParams {
    Particle* particles;
    int numParticles;
    int* d_startIndex;
    Entry* d_spatiacleLookat;
    int* d_isConverged;
};
__device__ inline float clamp(float x, float a, float b) {
    return fminf(fmaxf(x, a), b);
}
__device__ float poly6Kernel(const Vec2& r, float h) {
    float r2 = r.x * r.x + r.y * r.y;
    if (r2 >= h * h || h <= 0) return 0.0f;
    float h2 = h * h;
    float coefficient = 4.0f / (M_PI * pow(h, 4)); // 2D系数
    return coefficient * pow(h2 - r2, 3);
}
__device__ Vec2 PBFConstraintcuda::computeGradient(const Particle& pi, const Particle& pj, float h) {
    Vec2 r = pi.prediction - pj.prediction;
    float dist = r.length();
    if (dist <= 1e-5f || dist > h) return Vec2();
    float coefficient = -20.0f / (M_PI * pow(h, 3)); // 2D系数
    return r.normalize() * coefficient * pow(h - dist, 2);
}
__device__ float  PBFConstraintcuda::computeConstraint(const Particle& p, float restDensity) {
    return (p.density / restDensity)-1;
}
__device__ float PBFConstraintcuda::computeConstraint(const Particle& pi, const Particle& pj, float restDensity, float particleRadius) {
    // 密度约束
    float densityConstraint = (pi.density / restDensity) - 1.0f;

    // 重叠约束（当粒子间距小于2倍半径时触发）
    Vec2 offset = pi.prediction - pj.prediction;
    float dist = offset.length();
    float minDist = 2.0f * particleRadius;
    float overlapConstraint = (dist < minDist) ? (minDist - dist) / minDist : 0.0f;

    // 合并约束（权重可调）
    return densityConstraint + 0.5f * overlapConstraint;
}
__device__ float PBFConstraintcuda::computeRadius(const Particle& pi, const Particle& pj, float radius)
{
	return (pi.position - pj.position).length() - 2 * radius;
}
__device__ Vec2 PBFConstraintcuda::computeRadiusGradient(const Particle& pi, const Particle& pj, float radius) {
    Vec2 delta = pi.prediction - pj.prediction;
    float dist = delta.length();
    if (dist < 1e-5f) return Vec2();
    return delta / dist;
}
__device__ float PBFConstraintcuda::computeSurfaceTensionConstraint(const Particle& pi, const Particle& pj, float h) {
    Vec2 r = pi.prediction - pj.prediction;
    float dist = r.length();
    if (dist > h) return 0;
    return -poly6Kernel(r, h);
}
__device__ Vec2 PBFConstraintcuda::computeSurfaceTensionGradient(const Particle& pi, const Particle& pj, float h) {
    Vec2 r = pi.prediction - pj.prediction;
    float dist = r.length();
    if (dist <= 1e-5f || dist > h) return Vec2();
    float coefficient = -20.0f / (M_PI * pow(h, 3)); // 同Spiky梯度
    return r.normalize() * coefficient * pow(h - dist, 2);
}
__device__ float PBFConstraintcuda::computeViscosityConstraint(const Particle& pi, const Particle& pj) {
    Vec2 vij = pi.velocity - pj.velocity;
    Vec2 dx = pi.prediction - pj.prediction;
    return dx * vij;
}
__device__ Vec2 PBFConstraintcuda::computeViscosityGradient(const Particle& pi, const Particle& pj,float smoothingRadius)
{
    Vec2 vij = pi.velocity - pj.velocity;
    float len = vij.length();
    if (len <= 0.00001f || len > smoothingRadius) return Vec2();//距离过小视为重合
    return vij;
}
__device__ void PBFConstraintcuda::applyBoundaryConstraint(Particle& p, const Vec2& boundaryMin, const Vec2& boundaryMax, float bounceDamping) {
    // 处理X轴左边界
    if (p.prediction.X() < boundaryMin.X()) {
        p.prediction.x = boundaryMin.X();
        if (p.velocity.x < 0) {
            p.velocity.x *= -bounceDamping;
        }
    }
    // 处理X轴右边界
    else if (p.prediction.X() > boundaryMax.X()) {
        p.prediction.x = boundaryMax.X();
        if (p.velocity.X() > 0) {
            p.velocity.x *= -bounceDamping;
        }
    }

    // 处理Y轴下边界（假设Y向下为重力方向）
    if (p.prediction.Y() < boundaryMin.Y()) {
        p.prediction.y = boundaryMin.Y();
        if (p.velocity.Y() < 0) {
            p.velocity.y *= -bounceDamping;
        }
    }
    // 处理Y轴上边界
    else if (p.prediction.Y() > boundaryMax.Y()) {
        p.prediction.y = boundaryMax.Y();
        if (p.velocity.Y() > 0) {
            p.velocity.y *= -bounceDamping;
        }
    }

}
__device__ int getKeyFromHash(int Hash)
{
    return std::abs(Hash) % 2048;
}
__device__ void lookupNeighbors(const Entry* d_spatiacleLookat, const int* d_startIndex, int cellKey, int maxParticles, int* neighbors, int& count) 
{
    if (cellKey < 0) return;
    cellKey = getKeyFromHash(cellKey);

    int start = d_startIndex[cellKey];
    int end = d_startIndex[cellKey + 1];
    if (start >= end || start < 0 || end > maxParticles) return;

    for (int i = start; i < end && count < MAX_NEIGHBORS; ++i)
    {
        int neighborIdx = d_spatiacleLookat[i].index;
        if (neighborIdx < maxParticles)
        {
            neighbors[count++] = neighborIdx;
        }
    }
}
__device__ int HashCell(int x, int y)
{
    return (x * p1) ^ (y * p2);
}
__device__ int2 PositionToCellCoord(Vec2 point,float radius)
{
    int2 cell;
    cell.x = static_cast<int>(std::floor(point.X() / radius));
    cell.y= static_cast<int>(std::floor(point.Y() / radius));
    return cell;
}


__global__ void SolveOverlapConstraint(PBFSolverParams* d_params, float radius, Vec2* d_deltaPrediction) {
    extern __shared__ Particle sharedParticles[256];
    __shared__ int globalToShared[256];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int localIdx = threadIdx.x;

    if (idx < d_params->numParticles) {
        sharedParticles[localIdx] = d_params->particles[idx];
        globalToShared[localIdx] = idx;
    }
    __syncthreads();

    Particle& pi = d_params->particles[idx];
    Vec2 pos = pi.prediction;
    int2 cell = PositionToCellCoord(pos, 5);

    if (cell.x < 0 || cell.y < 0 || cell.x >= hashTableSize || cell.y >= hashTableSize) return;

    int neighborCells[9];
    int neighborCellCount = 0;
    for (int dx = -1; dx <= 1; ++dx) {
        for (int dy = -1; dy <= 1; ++dy) {
            int x = cell.x + dx;
            int y = cell.y + dy;
            if (x >= 0 && x < hashTableSize && y >= 0 && y < hashTableSize) {
                neighborCells[neighborCellCount++] = HashCell(x, y);
            }
        }
    }

    int neighbors[MAX_NEIGHBORS];
    int count = 0;
    for (int i = 0; i < neighborCellCount; ++i) {
        lookupNeighbors(d_params->d_spatiacleLookat, d_params->d_startIndex, neighborCells[i], d_params->numParticles, neighbors, count);
    }

    Vec2 dv(0, 0);
    float maxSeparation = 0.5f;
    for (int i = 0; i < count; ++i) {
        int j = neighbors[i];
        if (j == idx) continue;

        Particle pj = d_params->particles[j];
        Vec2 offset = pi.prediction - pj.prediction;
        float dist = offset.length();
        float minDist = 2.0f * radius;

        if (dist < minDist && dist > 1e-5f) {
            float alpha = (minDist - dist) / minDist;
            Vec2 dir = offset / dist;
            float lambda = 0.5f * alpha;
            Vec2 delta = dir * lambda * 0.5f;  // 各承担50%

            atomicAdd(&d_deltaPrediction[idx].x, delta.x);
            atomicAdd(&d_deltaPrediction[idx].y, delta.y);
            atomicAdd(&d_deltaPrediction[j].x, -delta.x);
            atomicAdd(&d_deltaPrediction[j].y, -delta.y);
        }
    }
}
__global__ void computeDensity(PBFSolverParams* params ,float smoothingRadius)
{
    extern __shared__ Particle sharedParticles[];
    __shared__ int globalToShared[256]; // 存储全局到共享内存的映射

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int localIdx = threadIdx.x;

	if (idx >= params->numParticles) return;
    if (idx < params->numParticles) {
        sharedParticles[localIdx] = params->particles[idx];
        globalToShared[localIdx] = idx;  // 建立映射
    }
    __syncthreads();

    Particle& pi = sharedParticles[localIdx];
    Vec2 pos = pi.prediction;
    int2 cell = PositionToCellCoord(pos, 20);
    // 检查当前cell是否有效
    if (cell.x < 0 || cell.y < 0 || cell.x >= hashTableSize || cell.y >= hashTableSize) return;

    int neighborCells[9];
    int neighborCellCount = 0;
    for (int dx = -1; dx <= 1; ++dx) {
        for (int dy = -1; dy <= 1; ++dy) {
            int x = cell.x + dx;
            int y = cell.y + dy;
            if (x >= 0 && x < hashTableSize && y >= 0 && y < hashTableSize) {
                neighborCells[neighborCellCount++] = HashCell(x, y);
            }
        }
    }
    int neighbors[MAX_NEIGHBORS];
    int count = 0;
    for (int i = 0; i < neighborCellCount; ++i) {
        lookupNeighbors(params->d_spatiacleLookat, params->d_startIndex, neighborCells[i], params->numParticles, neighbors, count);
    }
    for (int i = 0; i < count; ++i) {
		int j = neighbors[i];
		if (idx == j) continue;
		Particle pj(Vec2(0, 0));
		// 如果邻居粒子在当前 block 中，直接用共享内存
		if ((j / blockDim.x) == blockIdx.x)
			pj = sharedParticles[j % blockDim.x];
		else
			pj = params->particles[j];
        float r = (pi.position - pj.position).length(); // pi->position 访问
        if (r < smoothingRadius)
        {
            pi.density += pj.mass * 315.0f / (64.0f * M_PI * powf(smoothingRadius, 9.0f)) * powf((smoothingRadius * smoothingRadius - r * r), 8.0f);
        }
    }
}
__global__ void solveConstraints(PBFSolverParams* d_params, float resDensity,float tolerance, float epsilon, Vec2* deltaPrediction, Vec2* grads)
{
    extern __shared__ Particle sharedParticles[];
    __shared__ int globalToShared[256]; // 存储全局到共享内存的映射

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int localIdx = threadIdx.x;

    if (idx < d_params->numParticles) {
        sharedParticles[localIdx] = d_params->particles[idx];
        globalToShared[localIdx] = idx;  // 建立映射
    }
    __syncthreads();

    Particle& pi = d_params->particles[idx];
    Vec2* myGrads = grads + idx * d_params->numParticles;
    float C = PBFConstraintcuda::computeConstraint(pi, resDensity);
    if (C < 0)
    {
        return;
    }
    if (fabs(C) < tolerance) return;

    if (blockIdx.x == 0) {
        atomicExch(d_params->d_isConverged, 0);
    }
    if (idx >= d_params->numParticles) return;

    Vec2 pos = pi.prediction;
    
    int2 cell = PositionToCellCoord(pos, 20);
    // 检查当前cell是否有效
    if (cell.x < 0 || cell.y < 0 || cell.x >= hashTableSize || cell.y >= hashTableSize) return;

    int neighborCells[9];
    int neighborCellCount = 0;
    for (int dx = -1; dx <= 1; ++dx) {
        for (int dy = -1; dy <= 1; ++dy) {
            int x = cell.x + dx;
            int y = cell.y + dy;
            if (x >= 0 && x < hashTableSize && y >= 0 && y < hashTableSize) {
                neighborCells[neighborCellCount++] = HashCell(x, y);
            }
        }
    }
    int neighbors[MAX_NEIGHBORS];
    int count = 0;
    for (int i = 0; i < neighborCellCount; ++i) {
        lookupNeighbors(d_params->d_spatiacleLookat, d_params->d_startIndex, neighborCells[i], d_params->numParticles, neighbors, count);
    }

    Vec2 sumGrad2(0, 0);

    for (int i = 0; i < count; ++i) {
        int j = neighbors[i];
        if (idx == j) continue;

        Particle pj(Vec2(0, 0));

        bool foundInShared = false;
        for (int k = 0; k < blockDim.x; ++k) {
            if (globalToShared[k] == j) {
                pj = sharedParticles[k];
                foundInShared = true;
                break;
            }
        }
        if (!foundInShared) {
            pj = d_params->particles[j];
        }
        Vec2 grad = PBFConstraintcuda::computeGradient(pi, pj, 3.0f);
        myGrads[j] = grad;
        sumGrad2 += grad.cwiseProduct(grad);
    }

    float lambda = -C / (sumGrad2.X() + sumGrad2.Y() + epsilon);

    Vec2 delta(0.0f, 0.0f);
    for (int j = 0; j < count; ++j) {
        if (idx != neighbors[j]) {
            Vec2 grad = myGrads[neighbors[j]];
            delta += grad * lambda;

            atomicAdd(&deltaPrediction[j].x, -(grad * lambda).x);
            atomicAdd(&deltaPrediction[j].y, -(grad * lambda).y);

        }
    }
    atomicAdd(&deltaPrediction[idx].x, delta.X());
	atomicAdd(&deltaPrediction[idx].y, delta.Y());
}
__global__ void solveSurfaceTension(PBFSolverParams* d_params,float smoothingRadius, Vec2* deltaSurface)
{
    extern __shared__ Particle sharedParticles[];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int localIdx = threadIdx.x;

    if (idx >= d_params->numParticles) return;

    // 将当前粒子加载到共享内存
    sharedParticles[localIdx] = d_params->particles[idx];
    __syncthreads();

    Particle& pi = sharedParticles[localIdx];
    Vec2 pos = pi.prediction;

    int2 cell = PositionToCellCoord(pos, 10);
    int neighborCells[9] = {
        HashCell(cell.x - 1, cell.y - 1), HashCell(cell.x, cell.y - 1), HashCell(cell.x + 1, cell.y - 1),
        HashCell(cell.x - 1, cell.y), HashCell(cell.x, cell.y), HashCell(cell.x + 1, cell.y),
        HashCell(cell.x - 1, cell.y + 1), HashCell(cell.x, cell.y + 1), HashCell(cell.x + 1, cell.y + 1)
    };

    int neighbors[MAX_NEIGHBORS];
    int count = 0;

    for (int i = 0; i < 9; ++i) {
        lookupNeighbors(d_params->d_spatiacleLookat, d_params->d_startIndex, neighborCells[i], d_params->numParticles, neighbors, count);
    }

    Vec2 dv(0, 0);

    float totalWeight = 0.0f;
    for (int i = 0; i < count; ++i) {
        int j = neighbors[i];
        if (j == idx) continue;

        Particle pj(Vec2(0,0));
        if ((j / blockDim.x) == blockIdx.x) {
            pj = sharedParticles[j % blockDim.x];
        }
        else {
            pj = d_params->particles[j];
        }

        float weight = poly6Kernel(pi.prediction - pj.prediction, smoothingRadius);
        totalWeight += weight;

        float C = PBFConstraintcuda::computeSurfaceTensionConstraint(pi, pj, smoothingRadius);
        if (fabsf(C) < 0.00001f) continue;

        Vec2 grad = PBFConstraintcuda::computeSurfaceTensionGradient(pi, pj, 1.0f) * weight;
        dv += C * grad * 10.0f;

        // 正确的 atomicAdd
        atomicAdd(&(deltaSurface[j].x), -C * grad.x);
        atomicAdd(&(deltaSurface[j].y), -C * grad.y);

        // 仅当存在约束不满足时设置未收敛标志
        atomicExch(d_params->d_isConverged, 0);
    }

    // 归一化处理
    if (totalWeight > 0) {
        dv /= totalWeight;
    }

    atomicAdd(&(deltaSurface[idx].x), dv.x);
    atomicAdd(&(deltaSurface[idx].y), dv.y);
}
__global__ void solveViscosity(PBFSolverParams* d_params,Vec2* deltaViscosity)
{
    extern __shared__ Particle sharedParticles[256]; // 动态共享内存（用于存放线程块内的粒子）

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int localIdx = threadIdx.x;

    // 将当前粒子加载到共享内存
    sharedParticles[localIdx] = d_params->particles[idx];
    __syncthreads(); // 确保共享内存加载完毕

    Particle& pi = sharedParticles[localIdx];
    Vec2 pos = pi.prediction;

    int2 cell = PositionToCellCoord(pos, 20);
	if (cell.x <= 0 || cell.y <= 0) return; // 确保粒子在有效范围内
    int neighborCells[9] = {
        HashCell(cell.x - 1,cell.y - 1), HashCell(cell.x,cell.y - 1), HashCell(cell.x + 1,cell.y - 1),
        HashCell(cell.x - 1,cell.y), HashCell(cell.x,cell.y), HashCell(cell.x + 1,cell.y),
        HashCell(cell.x - 1,cell.y + 1),HashCell(cell.x ,cell.y + 1),HashCell(cell.x + 1,cell.y + 1)
    };

    int neighbors[MAX_NEIGHBORS];
    int count = 0;

    for (int i = 0; i < 9; ++i)
    {
        lookupNeighbors(d_params->d_spatiacleLookat, d_params->d_startIndex, neighborCells[i], d_params->numParticles, neighbors, count);
    }

    Vec2 dv(0, 0); // 局部累积的粘性校正
    for (int i = 0; i < count; ++i)
    {
        int j = neighbors[i];
        if (j == idx || j >= d_params->numParticles) continue;

        Particle pj(Vec2(0,0));
        // 如果邻居粒子在当前 block 中，直接用共享内存
        if ((j / blockDim.x) == blockIdx.x)
            pj = sharedParticles[j % blockDim.x];
        else
            pj = d_params->particles[j];

        float C = PBFConstraintcuda::computeViscosityConstraint(pi, pj);
        if (fabsf(C) < 1e-5f) continue;

        atomicExch(d_params->d_isConverged, 0);
        Vec2 grad = PBFConstraintcuda::computeViscosityGradient(pi, pj, 1.0f);

        // 累积当前粒子的 delta
        dv += C * grad;

        // 使用 atomicAdd 修改邻居粒子的 delta
        atomicAdd(&deltaViscosity[j].x, -C * grad.X());
        atomicAdd(&deltaViscosity[j].y, -C * grad.Y());
    }

    // 写回当前粒子的 deltaViscosity
    atomicAdd(&deltaViscosity[idx].x, dv.X());
    atomicAdd(&deltaViscosity[idx].y, dv.Y());
}
__global__ void applyBoundaryAndVelocityUpdate(Particle* particles, Vec2 boundaryMin, Vec2 boundaryMax, int numParticles, float dt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numParticles) {
       

        Vec2 newVelocity = (particles[idx].prediction - particles[idx].position) / dt;
        const float maxSpeed = 100.0f;

        if (newVelocity.length() > maxSpeed)
            newVelocity = newVelocity.normalize() * maxSpeed;

        particles[idx].velocity = newVelocity;
        PBFConstraintcuda::applyBoundaryConstraint(particles[idx], boundaryMin, boundaryMax, 0.8f);
        particles[idx].position = particles[idx].prediction;
    }
}
__global__ void updatePrediction(Particle* particles, Vec2* d_deltaPrediction,int numParticles) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < numParticles) {
        particles[idx].prediction += d_deltaPrediction[idx];
	}
}
__global__ void updateForce(Particle* particles, Vec2* d_deltaSurface, Vec2* d_deltaViscosity, int numParticles) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numParticles) {
        particles[idx].force = d_deltaSurface[idx] + d_deltaViscosity[idx];
        particles[idx].force += Vec2(0,1000.0f);
    }
}
__global__ void applyForceAndPrediction(Particle* particles, int numParticles,Vec2 gravity,float dt)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numParticles)
    {
        particles[idx].velocity += gravity * dt;
		particles[idx].prediction = particles[idx].position + particles[idx].velocity * dt;
    }
}
void solve(std::vector<Particle>& particles, float dt,std::vector<Entry> spatiacleLookat,std::vector<int> startIndex) {
    int numParticles = particles.size();
    int maxIterations = 8;
    ParticleGrid::getInstance().UpdateParticleLookat();
    Particle* d_particles;
    Entry* d_spatiacleLookat;
    int* d_startIndex;
    cudaMalloc(&d_particles, numParticles * sizeof(Particle));
    cudaMemcpy(d_particles, particles.data(), numParticles * sizeof(Particle), cudaMemcpyHostToDevice);
    cudaMalloc(&d_spatiacleLookat, sizeof(Entry) * numParticles);
    cudaMemcpy(d_spatiacleLookat, spatiacleLookat.data(), sizeof(Entry) * numParticles, cudaMemcpyHostToDevice);
    cudaMalloc(&d_startIndex, sizeof(int) * startIndex.size());
    cudaMemcpy(d_startIndex, startIndex.data(), sizeof(int) * startIndex.size(), cudaMemcpyHostToDevice);

    Vec2* d_deltaPrediction;

    Vec2* d_deltaSurface;
    Vec2* d_deltaViscosity;
    Vec2* d_grads;
    cudaMalloc(&d_grads, numParticles * numParticles * sizeof(Vec2));
    cudaMalloc(&d_deltaPrediction, numParticles * sizeof(Vec2));

    cudaMalloc(&d_deltaSurface, numParticles * sizeof(Vec2));
    cudaMalloc(&d_deltaViscosity, numParticles * sizeof(Vec2));
    cudaMemset(d_deltaPrediction, 0, numParticles * sizeof(Vec2));
    cudaMemset(d_deltaSurface, 0, numParticles * sizeof(Vec2));
    cudaMemset(d_deltaViscosity, 0, numParticles * sizeof(Vec2));
	cudaMemset(d_grads, 0, numParticles * numParticles * sizeof(Vec2));
	cudaMemset(d_deltaPrediction, 0, numParticles * sizeof(Vec2));


    // 收敛标志
    int h_isConverged;
    int* d_isConverged;

    cudaMalloc(&d_isConverged, sizeof(int));


    const int blockSize = 256;
    const int numBlocks = (numParticles + blockSize - 1) / blockSize;
   
    applyForceAndPrediction << <numBlocks, blockSize >> > (d_particles, numParticles,Vec2(0,100),dt);
    PBFSolverParams* params = (PBFSolverParams*)malloc(sizeof(PBFSolverParams));;
    params->particles = d_particles;
    params->numParticles = numParticles;
    params->d_startIndex = d_startIndex;
    params->d_spatiacleLookat = d_spatiacleLookat;
    params->d_isConverged = d_isConverged;

    PBFSolverParams* d_params;
    cudaMalloc(&d_params, sizeof(PBFSolverParams));
    cudaMemcpy(d_params, params, sizeof(PBFSolverParams), cudaMemcpyHostToDevice);

    for (int iter = 0; iter < maxIterations; ++iter) {
        // 设置收敛标志为 true
        h_isConverged = 1;
        cudaMemcpy(d_isConverged, &h_isConverged, sizeof(int), cudaMemcpyHostToDevice);
       
        computeDensity << <numBlocks, blockSize >> > (d_params, 5.0f);
        solveConstraints << <numBlocks, blockSize >> > (d_params,1000, 0.0001, 0.0001,d_deltaPrediction,d_grads);
       solveSurfaceTension << <numBlocks, blockSize >> > (d_params,20.0f,d_deltaSurface);
       solveViscosity << <numBlocks, blockSize >> > (d_params, d_deltaViscosity);
	   SolveOverlapConstraint << <numBlocks, blockSize>> > (d_params,8.0f, d_deltaPrediction);
	   updateForce << <numBlocks, blockSize >> > (
		   d_particles, d_deltaSurface, d_deltaViscosity, numParticles
		   );
       updatePrediction << <numBlocks, blockSize >> > (
            d_particles, d_deltaPrediction, numParticles
            );
        cudaMemcpy(&h_isConverged, d_isConverged, sizeof(int), cudaMemcpyDeviceToHost);
        if (h_isConverged == 1) break;

    }
    applyBoundaryAndVelocityUpdate << <numBlocks, blockSize >> > (
        d_particles, Vec2(1.0f, 1.0f), Vec2(800.0f, 450.0f),
        numParticles, dt
     );
    
    cudaDeviceSynchronize();

    cudaMemcpy(particles.data(), d_particles, numParticles * sizeof(Particle), cudaMemcpyDeviceToHost);
    // 清理
    cudaFree(d_particles);
    cudaFree(d_deltaPrediction);
    cudaFree(d_deltaSurface);
    cudaFree(d_deltaViscosity);
    cudaFree(d_isConverged);
    cudaFree(d_grads);
    cudaFree(d_spatiacleLookat);
    cudaFree(d_startIndex);
}