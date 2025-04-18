#include "Integrator.h"
void Integrator::step(std::vector<Particle>& particles, float ds, float mass) {
	for (auto& particle : particles) {
		particle.velocity += (particle.force / mass) * ds;

		particle.position += particle.velocity * ds;
	}
}