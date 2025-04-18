#include "SPH.h"

#define M_PI 3.14159265358979323846
float SPHSolver::poly6(float r, float h) {
	float hr2 = h * h - r * r;
	return (315.0f / (64.0f * M_PI * pow(h, 9))) * hr2 * hr2 * hr2;
}
float SPHSolver::spiky(float r, float h) {
	return (r < h) ? 15.0f / (M_PI * pow(h, 6)) * (h - r) * (h - r) * (h-r) : 0.0f;
}
float SPHSolver::spikyGradient(float r, float h)
{
	return (r < h) ? 15.0f / (M_PI * pow(h, 6)) * pow(h - r, 2) : 0.0f;
}
Vec2 SPHSolver::viscosityForce(const Particle& pi, const Particle& pj)
{
	Vec2 r = pi.position - pj.position;
	float dist = r.length();
	if (dist <= 0.00001f || dist > smoothingRadius) return Vec2();//距离过小视为重合
	float mu = viscosity * (pi.velocity - pj.velocity).length();
	return mu * poly6(dist, smoothingRadius) * r.normalize();
}
Vec2 SPHSolver::surfaceTensionForce(const Particle& pi, const Particle& pj)
{
	Vec2 r = pi.position - pj.position;
	float dist = r.length();
	if (dist <= 0.00001f || dist > smoothingRadius) return Vec2();//距离过小视为重合
	return -surfaceTension * spiky(dist, smoothingRadius) * r.normalize();
}