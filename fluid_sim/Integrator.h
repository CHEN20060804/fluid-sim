#pragma once
#include "Particle.h"
#include<vector>
class Integrator {
public:
	static void step(std::vector<Particle>& particles, float ds, float mass);
};