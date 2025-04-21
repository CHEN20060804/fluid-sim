#include "XPBD.h"
#define M_PI 3.14159265358979323846
Vec2 boundaryMin(50.0f, 50.0f);
Vec2 boundaryMax(400.0f, 250.0f);
float XPBDConstraint::computeConstraint(const Particle& pi) {
	return pi.density - restDensity;
}
Vec2 XPBDConstraint::computeGradient(const Particle& pi,const Particle& pj) {
	Vec2 r = pi.prediction - pj.prediction;
	float dist = r.length();
	if (dist <= 0.00001f || dist > smoothingRadius) return Vec2();//�����С��Ϊ�غ�
	return r.normalize() * (-45.0f / (M_PI * pow(smoothingRadius, 6))) * (pow(smoothingRadius - dist, 2));
}

float XPBDConstraint::computeSurfaceTensionConstraint(const Particle& pi, const Particle& pj, float restDistance) {
	return (pi.prediction - pj.prediction).length() - restDistance;
}
Vec2 XPBDConstraint::computeSurfaceTensionGradient(const Particle& pi, const Particle& pj)
{
    Vec2 r = pi.prediction - pj.prediction;
	float len = r.length();
	if (len <= 0.00001f || len > smoothingRadius) return Vec2();//�����С��Ϊ�غ�
    return r / len;
}

float XPBDConstraint::computeViscosityConstraint(const Particle& pi, const Particle& pj) {
	Vec2 vij = pi.velocity - pj.velocity;
	Vec2 dx = pi.prediction - pj.prediction;
    return dx * vij;
}
Vec2 XPBDConstraint::computeViscosityGradient(const Particle& pi, const Particle& pj)
{
	Vec2 vij = pi.velocity - pj.velocity;
    float len = vij.length();
	if (len <= 0.00001f || len > smoothingRadius) return Vec2();//�����С��Ϊ�غ�
    return vij;
}
void XPBDConstraint::applyBoundaryConstraint(Particle& p, const Vec2& boundaryMin, const Vec2& boundaryMax, float bounceDamping) {
    
        if (p.prediction.X() < boundaryMin.X()) {
            p.prediction.getX() = boundaryMin.X();
            p.velocity.getX() *= -bounceDamping;
        }
        else if (p.prediction.X() > boundaryMax.X()) {
            p.prediction.getX() = boundaryMax.X();
            p.velocity.getX() *= -bounceDamping;
        }
		if (p.prediction.Y() < boundaryMin.Y()) {
			p.prediction.getY() = boundaryMin.Y();
			p.velocity.getY() *= -bounceDamping;
		}
        else if (p.prediction.Y() > boundaryMax.Y()) {
            p.prediction.getY() = boundaryMax.Y();
            p.velocity.getY() *= -bounceDamping;
        }
}

void XPBDConstraint::solve(std::vector<Particle>& particles, float dt) {
    const int maxIterations = 10;  // ����������
    const float tolerance = 1e-5f;  // ������̶�

    // ��һ��Ԥ�⣺���ڵ�ǰ�ٶȽ���λ��Ԥ��
    for (auto& pi : particles) {
        pi.prediction = pi.position + pi.velocity * dt;
    }

    // ��ÿ��ʱ�䲽�У����Ƚ��ж��Լ������
    for (int iter = 0; iter < maxIterations; ++iter) {
        bool isConverged = true;

        // ����ÿ�����ӵ��ܶ�
        for (auto& pi : particles) {
            pi.density = computeDensityPrediction(pi, particles);
        }

        // �����ܶ�Լ��
        for (auto& pi : particles) {
            float C = computeConstraint(pi);  // �����ܶ�Լ��Υ����
            if (std::abs(C) < tolerance) continue;  // �������㹻С����������������

            Vec2 sumGrad2(0, 0);
            // ����ÿ�����ӵ�Լ���ݶȺ�������
            for (auto& pj : particles) {
                if (&pi == &pj) continue;

                Vec2 grad = computeGradient(pi, pj);
                sumGrad2 += grad.cwiseProduct(grad);  // �ݶȵ�ƽ����
            }

            // �����������ճ��� ��
            float lambda = C / (sumGrad2.length() + compliance);  // ���� ��

            // �������ӵ�λ��
            for (auto& pj : particles) {
                if (&pi == &pj) continue;

                Vec2 grad = computeGradient(pi, pj);
                pj.prediction += lambda * grad;  // ����λ�ã�ʹ��Ԥ��λ�ý���������
            }

            // ���Լ��û������������Ϊδ����
            if (std::abs(C) > tolerance) {
                isConverged = false;
            }
        }

        // �����������Լ��
        for (auto& pi : particles) {
            for (auto& pj : particles) {
                if (&pi == &pj) continue;

                float C_surface = computeSurfaceTensionConstraint(pi, pj, smoothingRadius);
                if (std::abs(C_surface) < tolerance) continue;  // �������㹻С����������������

                Vec2 grad_surface = computeSurfaceTensionGradient(pi, pj);
                pi.prediction += grad_surface * C_surface * compliance;  // ����λ��
            }
        }

        // ����ճ��Լ��
        for (auto& pi : particles) {
            for (auto& pj : particles) {
                if (&pi == &pj) continue;

                float C_viscosity = computeViscosityConstraint(pi, pj);
                if (std::abs(C_viscosity) < tolerance) continue;  // �������㹻С����������������

                Vec2 grad_viscosity = computeViscosityGradient(pi, pj);
                pi.prediction += grad_viscosity * C_viscosity * compliance;  // ����λ��
            }
        }

        // ����������ӵ�Լ���Ѿ����㣬����ǰ�˳�
        if (isConverged) {
            break;
        }
    }

    // �ڶ���Ԥ�⣺����Լ���������Ԥ��λ�ø���ʵ��λ��
    for (auto& pi : particles) {
        applyBoundaryConstraint(pi, boundaryMin, boundaryMax, 0.9);

        Vec2 newVelocity = (pi.prediction - pi.position) / dt;

        // ��������ٶȣ���ֹ˲ʱ����
        const float maxSpeed = 500.0f;
        if (newVelocity.length() > maxSpeed)
            newVelocity = newVelocity.normalize() * maxSpeed;

        pi.velocity = newVelocity;
        pi.position = pi.prediction;
    }
}
float XPBDConstraint::computeDensityPrediction(const Particle& pi, const std::vector<Particle>& particles) {
    float density = 0.0f;
    for (const auto& pj : particles) {
        float dist = (pi.prediction - pj.prediction).length();
        if (dist < smoothingRadius) {
            // ���� Poly6 �˺���
            float term = smoothingRadius * smoothingRadius - dist * dist;
            density += (315.0f / (64.0f * M_PI * pow(smoothingRadius, 9))) * term * term * term;
        }
    }
    return density;
}