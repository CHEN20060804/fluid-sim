#pragma once

#include <vector>
#include <numeric>
#include <execution>
#include "utility.h"
#include "Particle.h"
#include "ParticleGrid.h"

class SPHSolver {
public:
    // ģ�����
	SPHSolver(std::vector<Particle>& particles) :particles(particles),smoothingRadius(0.1f), viscosity(0.1f), surfaceTension(0.0728f), restDensity(1000.0f) {}
    float smoothingRadius = 3.0f;
    float viscosity = 0.1f;
    float surfaceTension = 0.0728f;
    float restDensity = 1.0f;

    // ���Ӽ��Ϻ�����
    std::vector<Particle>& particles;
  
    // ��Ҫ���躯��
    void simulateStep(float deltaTime);

    // ���ļ���
    Vec2 viscosityForce(const Particle& pi, const std::vector<Particle>& neighbors);
    Vec2 surfaceTensionForce(const Particle& pi, const std::vector<Particle>& neighbors);
    Vec2 computePressureForce(const Particle& pi, const std::vector<Particle>& particles);

    // �ܶ���ѹ������
    float computeDensity(const Particle& pi, const std::vector<Particle>& particles);
    float computeEveryPressure(float density);

private:
    // �˺���
    float poly6(float r, float h);
    float spiky(float r, float h);
    float spikyGradient(float r, float h);
};
