#include "Integrator.h"
//void Integrator::step(Particle &particle, float ds, float mass) {
//	
//		particle.velocity += (particle.force / mass) * ds;
//		particle.position += particle.velocity * ds;
//
//}
void Integrator::step(Particle& particle, float deltaTime,float mass) {
	// 假设粒子有 mass 属性，且已设定为 1.0f 或任意值
	float invMass = 1.0f;  // 防止除0

	// 半隐式欧拉积分（先更新速度，再更新位置）
	particle.velocity += particle.force * invMass * deltaTime;
	particle.position += particle.velocity * deltaTime;

	Boundary boundary;
	boundary.ResolveCollisions(particle.velocity, particle.position, 1.0f);

	// 更新 prediction（可选：若 prediction 仅用于预测）
	particle.prediction = particle.position;

	 // 0.5f 为阻尼系数


	// 可选：加阻尼，避免爆炸
	// particle.velocity *= 0.98f; // 轻微能量损失，防止粒子加速过猛
}