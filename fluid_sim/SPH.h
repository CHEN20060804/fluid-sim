#include "utility.h"
#include "Particle.h"
#include <execution>
#include "ParticleGird.h"
class SPHSolver {
public:
	SPHSolver(std::vector<Particle>& particlesRef) : particles(particlesRef) {}
	float smoothingRadius = 0.1f;
	float viscosity = 0.1f;
	float surfaceTension = 0.0728f;
	~SPHSolver() {
		particles.clear();
	};

	std::vector<Particle>& particles;
	Vec2 viscosityForce(const Particle& pi, const std::vector<Particle>& pj);
	Vec2 surfaceTensionForce(const Particle& pi, const std::vector<Particle>& pj);

	void simulateStep(float delteTime);
private:
	float poly6(float r, float h);
	float spikyGradient(float r, float h);
	float spiky(float r, float h);
	ParticleGird particleGird{particles};
};
