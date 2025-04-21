#include "SPH.h"
#include "Integrator.h"
#include <fstream>
const int gravity = 10;
#define M_PI 3.14159265358979323846
float SPHSolver::poly6(float r, float h)
{
	float hr2 = h * h - r * r;
	return (315.0f / (64.0f * M_PI * pow(h, 9))) * hr2 * hr2 * hr2;
}

float SPHSolver::computeDensity(const Particle& pi, const std::vector<Particle>& particles)//计算密度
{
	float density = 0.0f;
	for (const Particle& pj : particles) {
		Vec2 r = pi.prediction - pj.prediction;
		float dist = r.length();
		if (dist < smoothingRadius)
			density +=poly6(dist, smoothingRadius);//质量为1
	}
	return density;
}

float SPHSolver::computeEveryPressure(float density)
{
	float stiffness = 3.0f;
	return stiffness * std::max(0.0f,(density - restDensity));
}

Vec2 SPHSolver::computePressureForce(const Particle& pi, const std::vector<Particle>& particles)
{
	Vec2 totalForce(0.0f, 0.0f);

	for (const Particle& pj : particles) 
	{
		if (&pi == &pj) continue; // 不和自己算力

		Vec2 r = pi.prediction - pj.prediction;
		float dist = r.length();
		if (dist <= 0.000001f || dist > smoothingRadius) continue;

		float pij = (pi.pressure + pj.pressure) / 2.0f;
		float grad = spikyGradient(dist, smoothingRadius);

		float pjDensity = std::max(pj.density, 0.0001f);
		Vec2 force = - pij / pjDensity * grad * r.normalize();
		totalForce += force;
	}

	return totalForce;
}

float SPHSolver::spiky(float r, float h) {
	return (r < h) ? 15.0f / (M_PI * pow(h, 6)) * (h - r) * (h - r) * (h-r) : 0.0f;
}
float SPHSolver::spikyGradient(float r, float h)
{
	return (r < h) ? 15.0f / (M_PI * pow(h, 6)) * pow(h - r, 2) : 0.0f;
}
Vec2 SPHSolver::viscosityForce(const Particle& pi, const std::vector<Particle>& neighbors)
{
	Vec2 force(0.0f, 0.0f);
	for (const Particle& pj : neighbors)
	{
		if (&pi == &pj) continue;

		Vec2 r = pi.prediction - pj.prediction;
		float dist = r.length();
		if (dist <= 0.00001f || dist > smoothingRadius) continue;

		float weight = poly6(dist, smoothingRadius);
		force += viscosity * (pj.velocity - pi.velocity) * weight;
	}

	return force;
}
Vec2 SPHSolver::surfaceTensionForce(const Particle& pi, const std::vector<Particle>& pj)
{
	for (int i = 0; i < pj.size(); i++)
	{
		if (pi.position == pj[i].position) continue;

		Vec2 r = pi.position - pj[i].position;
		float dist = r.length();
		if (dist <= 0.00001f || dist > smoothingRadius) return Vec2();//距离过小视为重合

		return -surfaceTension * spiky(dist, smoothingRadius) * r.normalize();
	}
	}
	
void SPHSolver::simulateStep(float delteTime)
{
	std::vector<size_t> indices(particles.size());
	std::iota(indices.begin(), indices.end(), 0);  // 填充为 0, 1, 2, ..., N-1

	std::for_each(std::execution::par, indices.begin(), indices.end(), [&](size_t i) {
		Particle& pi = particles[i];
		pi.force = Vec2(0.0f, 0.0f);
		pi.velocity += Vec2(0, 1) * gravity * delteTime;// 重置力
		pi.prediction = pi.position + pi.velocity * delteTime;//目前测试，暂时改为position
		std::wofstream log("prrdiction.txt", std::ios::app);
		log << L"particle[" << i << L"] prediction = " << pi.prediction.getX() << " " << pi.prediction.getY() << L"\n";
		});
	//ParticleGrid::getInstance().UpdateParticleLookat();

	/*Parallel.For(0, numParticles.i = >
	{
		densities[i] = CalculateDensity(predictedPositions[i]);
	});*/
	std::for_each(std::execution::par, indices.begin(), indices.end(), [&](size_t i) {
		//std::vector<Particle> other = ParticleGrid::getInstance().ForeachPointWithinRadius(particles[i].prediction, this->smoothingRadius);
		/*std::wofstream log("neighbor_debug.txt", std::ios::app);
		log << L"particle[" << i << L"] neighbor count = " << other.size() << L"\n";*/
		particles[i].density = computeDensity(particles[i], particles);
	});

	std::for_each(std::execution::par, indices.begin(), indices.end(), [&](size_t i) {
		    //std::vector<Particle> other = ParticleGrid::getInstance().ForeachPointWithinRadius(particles[i].prediction, this->smoothingRadius);
			//particles[i].force += viscosityForce(particles[i], other);
			particles[i].pressure = computeEveryPressure(particles[i].density);
			std::wofstream log("debug_log.txt", std::ios::app); // 追加写入
			
			    
			particles[i].force += computePressureForce(particles[i], particles);
			log << L"particle[" << i << L"] pressure=("
				<< particles[i].pressure << L" "
				<< particles[i].force.getX() << L" "
				<< particles[i].force.getY() << L" "
				<< particles[i].density << L"\n ";
			//particles[i].force += surfaceTensionForce(particles[i], other);
		});

	std::for_each(std::execution::par, indices.begin(), indices.end(), [&](size_t i) {
		Integrator::step(particles[i], delteTime, 1.0f);
		});


}