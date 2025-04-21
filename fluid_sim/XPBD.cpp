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
	if (dist <= 0.00001f || dist > smoothingRadius) return Vec2();//距离过小视为重合
	return r.normalize() * (-45.0f / (M_PI * pow(smoothingRadius, 6))) * (pow(smoothingRadius - dist, 2));
}

float XPBDConstraint::computeSurfaceTensionConstraint(const Particle& pi, const Particle& pj, float restDistance) {
	return (pi.prediction - pj.prediction).length() - restDistance;
}
Vec2 XPBDConstraint::computeSurfaceTensionGradient(const Particle& pi, const Particle& pj)
{
    Vec2 r = pi.prediction - pj.prediction;
	float len = r.length();
	if (len <= 0.00001f || len > smoothingRadius) return Vec2();//距离过小视为重合
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
	if (len <= 0.00001f || len > smoothingRadius) return Vec2();//距离过小视为重合
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
    const int maxIterations = 10;  // 最大迭代次数
    const float tolerance = 1e-5f;  // 误差容忍度

    // 第一次预测：基于当前速度进行位置预测
    for (auto& pi : particles) {
        pi.prediction = pi.position + pi.velocity * dt;
    }

    // 在每个时间步中，首先进行多次约束解算
    for (int iter = 0; iter < maxIterations; ++iter) {
        bool isConverged = true;

        // 计算每个粒子的密度
        for (auto& pi : particles) {
            pi.density = computeDensityPrediction(pi, particles);
        }

        // 处理密度约束
        for (auto& pi : particles) {
            float C = computeConstraint(pi);  // 计算密度约束违反量
            if (std::abs(C) < tolerance) continue;  // 如果误差足够小，可以跳过该粒子

            Vec2 sumGrad2(0, 0);
            // 计算每对粒子的约束梯度和修正量
            for (auto& pj : particles) {
                if (&pi == &pj) continue;

                Vec2 grad = computeGradient(pi, pj);
                sumGrad2 += grad.cwiseProduct(grad);  // 梯度的平方和
            }

            // 计算拉格朗日乘子 λ
            float lambda = C / (sumGrad2.length() + compliance);  // 计算 λ

            // 更新粒子的位置
            for (auto& pj : particles) {
                if (&pi == &pj) continue;

                Vec2 grad = computeGradient(pi, pj);
                pj.prediction += lambda * grad;  // 调整位置（使用预测位置进行修正）
            }

            // 如果约束没有收敛，则标记为未收敛
            if (std::abs(C) > tolerance) {
                isConverged = false;
            }
        }

        // 处理表面张力约束
        for (auto& pi : particles) {
            for (auto& pj : particles) {
                if (&pi == &pj) continue;

                float C_surface = computeSurfaceTensionConstraint(pi, pj, smoothingRadius);
                if (std::abs(C_surface) < tolerance) continue;  // 如果误差足够小，可以跳过该粒子

                Vec2 grad_surface = computeSurfaceTensionGradient(pi, pj);
                pi.prediction += grad_surface * C_surface * compliance;  // 修正位置
            }
        }

        // 处理粘性约束
        for (auto& pi : particles) {
            for (auto& pj : particles) {
                if (&pi == &pj) continue;

                float C_viscosity = computeViscosityConstraint(pi, pj);
                if (std::abs(C_viscosity) < tolerance) continue;  // 如果误差足够小，可以跳过该粒子

                Vec2 grad_viscosity = computeViscosityGradient(pi, pj);
                pi.prediction += grad_viscosity * C_viscosity * compliance;  // 修正位置
            }
        }

        // 如果所有粒子的约束已经满足，则提前退出
        if (isConverged) {
            break;
        }
    }

    // 第二次预测：基于约束修正后的预测位置更新实际位置
    for (auto& pi : particles) {
        applyBoundaryConstraint(pi, boundaryMin, boundaryMax, 0.9);

        Vec2 newVelocity = (pi.prediction - pi.position) / dt;

        // 限制最大速度，防止瞬时跳出
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
            // 常用 Poly6 核函数
            float term = smoothingRadius * smoothingRadius - dist * dist;
            density += (315.0f / (64.0f * M_PI * pow(smoothingRadius, 9))) * term * term * term;
        }
    }
    return density;
}