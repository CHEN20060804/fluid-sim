#pragma once
#include "Particle.h"
#include <vector>

class PBFConstraint {
public:
	float restDensity = 500.0f;
	float smoothingRadius = 1.0f;
	float boundary = -1.0f;
	float epsilon = 1e-6f;
	void solve(std::vector<Particle>& particles, float dt);

private:
	Vec2 boundaryMin1 = Vec2(10.0f, 10.0f );
	Vec2 boundaryMax1 = Vec2(360.0f, 210.0f);
	float computeDensityPrediction(const Particle& pi, const std::vector<Particle>& particles);

	float computeConstraint(const Particle& pi);
	float computeSurfaceTensionConstraint(const Particle& pi, const Particle& pj, float restDensity);
	float computeViscosityConstraint(const Particle& pi, const Particle& pj);

	Vec2 computeGradient(const Particle& pi, const Particle& pj);
	Vec2 computeSurfaceTensionGradient(const Particle& pi, const Particle& pj);
	Vec2 computeViscosityGradient(const Particle& pi, const Particle& pj);
	void applyBoundaryConstraint(Particle& p, const Vec2& boundaryMin, const Vec2& boundaryMax, float bounceDamping);
};