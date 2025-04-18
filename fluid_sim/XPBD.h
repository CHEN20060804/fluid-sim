#pragma once
#include "Particle.h"
#include <vector>

class XPBDConstraint {
public:
	float compliance = 0.0001f;
	float restDensity = 1000.0f;
	float smoothingRadius = 0.1f;
	float boundary = -1.0f;

	void solve(std::vector<Particle>& particles, float dt);
private:
	float computeConstraint(const Particle& pi);
	Vec2 computeGradient(const Particle& pi,const Particle& pj);
};