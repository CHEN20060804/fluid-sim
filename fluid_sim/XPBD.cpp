#include "XPBD.h"
#define M_PI 3.14159265358979323846
float XPBDConstraint::computeConstraint(const Particle& pi) {
	return pi.density - restDensity - 1.0f;
}
Vec2 XPBDConstraint::computeGradient(const Particle& pi,const Particle& pj) {
	Vec2 r = pi.position - pj.position;
	float dist = r.length();
	if (dist <= 0.00001f || dist > smoothingRadius) return Vec2();//�����С��Ϊ�غ�
	return r.normalize() * (-45.0f / (M_PI * pow(smoothingRadius, 6))) * (pow(smoothingRadius - dist, 2));
}
void XPBDConstraint::solve(std::vector<Particle>& particles, float dt) {
    const int maxIterations = 10;  // ����������
    const float tolerance = 1e-5f;  // ������̶�

    // ��ÿ��ʱ�䲽�У����Ƚ��ж��Լ������
    for (int iter = 0; iter < maxIterations; ++iter) {
        bool isConverged = true;

        // ÿ�ε���ʱ������������
        for (auto& pi : particles) {
            float C = computeConstraint(pi);  // ����Լ��Υ����
            if (std::abs(C) < tolerance) continue;  // �������㹻С����������������

            Vec2 sumGrad2(0, 0);
            Vec2 deltaP(0, 0);

            // ����ÿ�����ӵ�Լ���ݶȺ�������
            for (auto& pj : particles) {
                if (&pi == &pj) continue;

                Vec2 grad = computeGradient(pi, pj);
                sumGrad2 += grad.cwiseProduct(grad);  // �ݶȵ�ƽ����
                deltaP += grad * (C / sumGrad2.length());  // λ��������
            }

            // �����������ճ��� ��
            float lambda = C / (sumGrad2.length() + compliance);  // ���� ��

            // �������ӵ�λ��
            for (auto& pj : particles) {
                if (&pi == &pj) continue;

                Vec2 grad = computeGradient(pi, pj);
                pj.position += lambda * grad;  // ����λ��
            }
            pi.position -= lambda * deltaP;  // ���� pi ��λ��

            // ���Լ��û������������Ϊδ����
            if (std::abs(C) > tolerance) {
                isConverged = false;
            }
        }
        // ����������ӵ�Լ���Ѿ����㣬����ǰ�˳�
        if (isConverged) {
            break;
        }
    }

    // ����߽�����
    applyBoundaryConditions(particles, boundary);
}