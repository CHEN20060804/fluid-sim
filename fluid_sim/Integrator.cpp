#include "Integrator.h"
//void Integrator::step(Particle &particle, float ds, float mass) {
//	
//		particle.velocity += (particle.force / mass) * ds;
//		particle.position += particle.velocity * ds;
//
//}
void Integrator::step(Particle& particle, float deltaTime,float mass) {
	// ���������� mass ���ԣ������趨Ϊ 1.0f ������ֵ
	float invMass = 1.0f;  // ��ֹ��0

	// ����ʽŷ�����֣��ȸ����ٶȣ��ٸ���λ�ã�
	particle.velocity += particle.force * invMass * deltaTime;
	particle.position += particle.velocity * deltaTime;

	Boundary boundary;
	boundary.ResolveCollisions(particle.velocity, particle.position, 1.0f);

	// ���� prediction����ѡ���� prediction ������Ԥ�⣩
	particle.prediction = particle.position;

	 // 0.5f Ϊ����ϵ��


	// ��ѡ�������ᣬ���ⱬը
	// particle.velocity *= 0.98f; // ��΢������ʧ����ֹ���Ӽ��ٹ���
}