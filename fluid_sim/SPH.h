#include "utility.h"
#include "Particle.h"
#include <vector>
class SPHSolver {
public:
	float smoothingRadius = 0.1f;
	float viscosity = 0.1f;
	float surfaceTension = 0.0728f;
	float restDensity = 1000.0f;

	Vec2 viscosityForce(const Particle& pi, const Particle& pj);
	Vec2 surfaceTensionForce(const Particle& pi, const Particle& pj);
	float computeDensity(const Particle& pi, const std::vector<Particle>& particles);
	Vec2 computePressureForce(const Particle& pi, const std::vector<Particle>& particles);
	void updateOneParticle(Particle& pi, const std::vector<Particle>& particles);
	float computeEveryPressure(float density);

private:
	float poly6(float r, float h);
	float spikyGradient(float r, float h);
	float spiky(float r, float h);
};
