#define NOMINMAX
#include "SPH.h"
#include <fstream>
#include <cmath>
#include <algorithm> 
#include "Boundary.h"
const int gravity = 80;
#define M_PI 3.14159265358979323846
//核函数
float SPHSolver::poly6(float r, float h)
{
	float hr2 = h * h - r * r;
	return (315.0f / (64.0f * M_PI * pow(h, 9))) * hr2 * hr2 * hr2;
}
float SPHSolver::spiky(float r, float h) {
	return (r < h) ? 15.0f / (M_PI * pow(h, 6)) * (h - r) * (h - r) * (h - r) : 0.0f;
}
float SPHSolver::spikyGradient(float r, float h)
{
	return (r < h) ? 15.0f / (M_PI * pow(h, 6)) * pow(h - r, 2) : 0.0f;
}
float SPHSolver::computeDensity(const Particle& pi, std::vector<std::reference_wrapper<Particle>> particles)
{
	float density = 0.0f;
	for (const Particle& pj : particles) {
		Vec2 r = pi.prediction - pj.prediction;
		float dist = r.length();
		if (dist < smoothingRadius)
			density += pj.mass * poly6(dist, smoothingRadius);//质量为1
	}
	return density;
}
float SPHSolver::computeDensityForPBF(const Particle& pi, std::vector<Particle> particles)
{
	float density = 0.0f;
	for (const Particle& pj : particles) {
		Vec2 r = pi.prediction - pj.prediction;
		float dist = r.length();
		if (dist < smoothingRadius)
			density += pj.mass * poly6(dist, smoothingRadius);//质量为1
	}
	return density;
}
// 计算压力
float SPHSolver::computeEveryPressure(float density)
{

	const float stiffness = 1000.0f;  
	const float gamma = 7.0f;        
	const float eps = 1e-5f;

	if (density < eps)
		return 0.0f;

	return stiffness * (pow(density / restDensity, gamma) - 1.0f);
}
Vec2 SPHSolver::computePressureForce(const Particle& pi, std::vector<std::reference_wrapper<Particle>> particles)
{
	Vec2 totalForce(0.0f, 0.0f);

	const float minDistance = 1.0f;         
	const float repulsionStrength = 50.0f;  // 排斥力系数
	const float eps = 1e-5f;

	for (const Particle& pj : particles)
	{
		if (&pi == &pj) continue;

		Vec2 r = pi.prediction - pj.prediction;
		float dist = r.length();
		if (dist < eps || dist > smoothingRadius) continue;

		Vec2 dir = (dist > eps) ? (r / dist) : Vec2(0.0f, 0.0f); // 归一化

		float grad = spikyGradient(dist, smoothingRadius);

		float pij = (pi.pressure + pj.pressure) * 0.5f;
		float pjDensity = std::max(pj.density, 0.0001f);
		Vec2 force = -pij * pj.mass / pjDensity * grad * dir;
		totalForce += force;

		// 添加额外的软性排斥力，避免粒子距离小于 minDistance
		if (dist < minDistance)
		{
			float overlap = minDistance - dist;
			Vec2 repelForce = repulsionStrength * overlap * dir;
			totalForce += repelForce;
		}
	}

	return totalForce * 8.0f;
}
//粘力
Vec2 SPHSolver::viscosityForce(const Particle& pi, std::vector<std::reference_wrapper<Particle>> neighbors)
{
	Vec2 force(0.0f, 0.0f);
	for (const Particle& pj : neighbors)
	{
		if (&pi == &pj) continue;

		Vec2 r = pi.prediction - pj.prediction;
		float dist = r.length();
		if (dist <= 0.00001f || dist > smoothingRadius) continue;

		float weight = poly6(dist, smoothingRadius);
		force += viscosity * pj.mass * (pj.velocity - pi.velocity) * weight;
	}

	return force;
}
Vec2 SPHSolver::XSPHVelocityCorrection(const Particle& pi, std::vector<std::reference_wrapper<Particle>> neighbors)
{
	Vec2 correction(0.0f, 0.0f);
	for (const Particle& pj : neighbors)
	{
		if (&pi == &pj) continue;
		Vec2 r = pi.prediction - pj.prediction;
		float dist = r.length();
		if (dist <= 0.00001f || dist > smoothingRadius) continue;

		float weight = poly6(dist, smoothingRadius);
		correction += (pj.mass / pj.density) * (pj.velocity - pi.velocity) * weight;
	}

	return viscosity * correction;
}
//表面张力
Vec2 SPHSolver::surfaceTensionForce(const Particle& pi, std::vector<std::reference_wrapper<Particle>> neighbors)
{
	Vec2 totalForce(0.0f, 0.0f);
	for (const Particle& pj : neighbors)
	{
		if (pi.position == pj.position) continue;
		Vec2 r = pi.position - pj.position;
		float dist = r.length();

		if (dist <= 0.00001f || dist > smoothingRadius) continue;

		// 表面张力力公式 -σ * W(r, h) * r̂
		totalForce += -surfaceTension * pj.mass * spiky(dist, smoothingRadius) * r.normalize();
	}
	return totalForce * surfaceTension;
}

void SPHSolver::resolveParticleOverlap(float minDist) {
	float maxSeparation = 0.5f;   // 单次最大分离距离
	float restitution = 0.86f;     // 弹性恢复系数

	size_t count = particles.size();

	for (size_t i = 0; i < count; ++i) {
		Particle& pi = particles[i];

		for (size_t j = i + 1; j < count; ++j) {
			Particle& pj = particles[j];

			Vec2 delta = pi.position - pj.position;
			float dist = delta.length();

			if (dist < minDist && dist > 1e-5f) {
				Vec2 dir = delta / dist; // normalize
				float overlap = minDist - dist;
				float separation = std::min(overlap * 0.5f, maxSeparation);

				pi.position += dir * separation;
				pj.position -= dir * separation;

				Vec2 relativeVelocity = pi.velocity - pj.velocity;
				float vAlongNormal = relativeVelocity * dir;

				if (vAlongNormal < 0.0f) {
					float impulseMag = -(1 + restitution) * vAlongNormal * 0.5f;
					Vec2 impulse = dir * impulseMag;
					pi.velocity += impulse;
					pj.velocity -= impulse;
				}
			}
		}
	}
}

//模拟
void SPHSolver::simulateStep(float deltaTime)
{

	std::vector<size_t> indices(particles.size());
	std::iota(indices.begin(), indices.end(), 0);

	std::for_each(std::execution::par, indices.begin(), indices.end(), [&](size_t i) {
		Particle& pi = particles[i];
		pi.velocity += Vec2(0, 1) * gravity * deltaTime * pi.mass;
		pi.prediction = pi.position + pi.velocity * deltaTime;
		});

	ParticleGrid::getInstance().UpdateParticleLookat();
	//密度
	std::for_each(std::execution::par, indices.begin(), indices.end(), [&](size_t i) {
		auto neighbors = ParticleGrid::getInstance().ForeachPointWithinRadius(particles[i].prediction, this->smoothingRadius);
		particles[i].density = computeDensity(particles[i], neighbors);
		});

	//力
	std::for_each(std::execution::par, indices.begin(), indices.end(), [&](size_t i) {
		auto neighbors = ParticleGrid::getInstance().ForeachPointWithinRadius(particles[i].prediction, this->smoothingRadius);
		//粘力，通过速度表达（Xsph）
		particles[i].velocity += XSPHVelocityCorrection(particles[i], neighbors);
		particles[i].pressure = computeEveryPressure(particles[i].density);

		// 计算压力力和表面张力
		particles[i].force += computePressureForce(particles[i], neighbors);
		particles[i].force += surfaceTensionForce(particles[i], neighbors);
		});
	// 更新粒子
	const int substeps = 20;
	float subDelta = deltaTime / substeps;

	for (int step = 0; step < substeps; ++step) {
		std::for_each(std::execution::par, indices.begin(), indices.end(), [&](size_t i) {
			float invMass = 1.0f / particles[i].mass;

			particles[i].velocity += particles[i].force * invMass * subDelta;
			particles[i].position += particles[i].velocity * subDelta;
			particles[i].force = Vec2(0.0f, 0.0f);

			boundary.ResolveCollisions(particles[i],0.8);
			});
	}

	resolveParticleOverlap(8.0f);
}
