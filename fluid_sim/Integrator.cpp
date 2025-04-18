#include "Integrator.h"
void Integrator::step(Particle particle, float ds, float mass) {
	
		particle.velocity += (particle.force / mass) * ds;
		particle.position += particle.velocity * ds;

}