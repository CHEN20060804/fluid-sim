#pragma once
#include "Particle.h"
#include <vector>

class XPBDConstraint {
public:
	float compliance = 0.0001f;
	float restDensity = 10000.0f;
	float smoothingRadius = 3.0f;
	float boundary = -1.0f;

	void solve(std::vector<Particle>& particles, float dt);
	
private:
	float computeDensityPrediction(const Particle& pi, const std::vector<Particle>& particles);

	float computeConstraint(const Particle& pi);
	float computeSurfaceTensionConstraint(const Particle& pi, const Particle& pj, float restDensity);
	float computeViscosityConstraint(const Particle& pi, const Particle& pj);

	Vec2 computeGradient(const Particle& pi,const Particle& pj);
	Vec2 computeSurfaceTensionGradient(const Particle& pi, const Particle& pj);
	Vec2 computeViscosityGradient(const Particle& pi, const Particle& pj);
	void applyBoundaryConstraint(Particle& p, const Vec2& boundaryMin, const Vec2& boundaryMax, float bounceDamping);
};