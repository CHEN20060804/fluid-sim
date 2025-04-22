#pragma once
#include "Particle.h"
#include<vector>
#include "Boundary.h"
class Integrator {
public:
	static void step(Particle &particle, float ds, float mass);
};