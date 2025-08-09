#pragma once
#include <vector>
#include <numeric>
#include <execution>
#include "utility.h"
#include "Particle.h"
#include "ParticleGrid.h"
#include "Boundary.h"
class SPHSolver {
public:

    SPHSolver(std::vector<Particle>& particles) :particles(particles), smoothingRadius(3.0f), viscosity(0.01f), surfaceTension(6.28f), restDensity(500.0f) {}
    float smoothingRadius;
    float viscosity;
    float surfaceTension;
    float restDensity;

    std::vector<Particle>& particles;

    void simulateStep(float deltaTime);

    // 力的计算
    Vec2 viscosityForce(const Particle& pi, std::vector<std::reference_wrapper<Particle>> neighbors);
    Vec2 surfaceTensionForce(const Particle& pi, std::vector<std::reference_wrapper<Particle>> neighbors);
    void resolveParticleOverlap(float minDist);
    Vec2 computePressureForce(const Particle& pi, std::vector<std::reference_wrapper<Particle>> particles);
    //影响位置
    float computeDensity(const Particle& pi, std::vector<std::reference_wrapper<Particle>> particles);
    float computeDensityForPBF(const Particle& pi, std::vector<Particle> particles);
    float computeEveryPressure(float density);
    Vec2 XSPHVelocityCorrection(const Particle& pi, std::vector<std::reference_wrapper<Particle>> neighbors);

private:
    Boundary boundary;
    float poly6(float r, float h);
    float spiky(float r, float h);
    float spikyGradient(float r, float h);
    std::vector<std::mutex> particleLocks = std::vector<std::mutex>(6000);
};