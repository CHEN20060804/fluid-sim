#pragma once

#include <vector>
#include <numeric>
#include <execution>
#include "utility.h"
#include "Particle.h"
#include "ParticleGrid.h"

class SPHSolver {
public:
    // 模拟参数
    float smoothingRadius = 0.1f;
    float viscosity = 0.1f;
    float surfaceTension = 0.0728f;
    float restDensity = 1000.0f;

    // 粒子集合和网格
    std::vector<Particle> particles;
    ParticleGrid particleGird;

    // 主要步骤函数
    void simulateStep(float deltaTime);

    // 力的计算
    Vec2 viscosityForce(const Particle& pi, const std::vector<Particle>& neighbors);
    Vec2 surfaceTensionForce(const Particle& pi, const std::vector<Particle>& neighbors);
    Vec2 computePressureForce(const Particle& pi, const std::vector<Particle>& particles);

    // 密度与压力计算
    float computeDensity(const Particle& pi, const std::vector<Particle>& particles);
    float computeEveryPressure(float density);

private:
    // 核函数
    float poly6(float r, float h);
    float spiky(float r, float h);
    float spikyGradient(float r, float h);
};
