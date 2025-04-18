#include "utility.h"
#include "Particle.h"
class SPHSolver {
public:
	SPHSolver(std::vector<Particle>& particlesRef) : particles(particlesRef) {}
	float smoothingRadius = 0.1f;
	float viscosity = 0.1f;
	float surfaceTension = 0.0728f;

	Vec2 viscosityForce(const Particle& pi, const Particle& pj);
	Vec2 surfaceTensionForce(const Particle& pi, const Particle& pj);

	void simulateStep(float delteTime);
private:
	float poly6(float r, float h);
	float spikyGradient(float r, float h);
	float spiky(float r, float h);
	ParticleGird particleGird{particles};
};
