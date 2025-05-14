#include "utility.h"
#include <vector>
#include <iostream>
#include "Particle.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "ParticleGrid.h"
namespace PBFConstraintcuda {
    __device__ inline float computeConstraint(const Particle& pi, float restDensity);
    __device__ inline float computeSurfaceTensionConstraint(const Particle& pi, const Particle& pj, float restDensity);
    __device__ inline float computeViscosityConstraint(const Particle& pi, const Particle& pj);
    __device__ inline float computeRadius(const Particle& pi, const Particle& pj, float radius);
    __device__ inline Vec2 computeGradient(const Particle& pi, const Particle& pj, float smoothingRadius);
    __device__ inline Vec2 computeRadiusGradient(const Particle& pi, const Particle& pj, float radius);
    __device__ inline float computeConstraint(const Particle& pi, const Particle& pj, float restDensity, float particleRadius);
    __device__ inline Vec2 computeSurfaceTensionGradient(const Particle& pi, const Particle& pj, float smoothingRadius);
    __device__ inline Vec2 computeViscosityGradient(const Particle& pi, const Particle& pj, float smoothingRadius);
    __device__ inline void applyBoundaryConstraint(Particle& p, const Vec2& boundaryMin, const Vec2& boundaryMax, float bounceDamping);
}
void solve(std::vector<Particle>& particles, float dt,std::vector<Entry> spatiacleLookat, std::vector<int> startIndex);