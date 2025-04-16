#include "XPBD.h"
#define M_PI 3.14159265358979323846
float XPBDConstraint::computeConstraint(const Particle& pi) {
	return pi.density - restDensity - 1.0f;
}
Vec2 XPBDConstraint::computeGradient(const Particle& pi,const Particle& pj) {
	Vec2 r = pi.position - pj.position;
	float dist = r.length();
	if (dist <= 0.00001f || dist > smoothingRadius) return Vec2();//距离过小视为重合
	return r.normalize() * (-45.0f / (M_PI * pow(smoothingRadius, 6))) * (pow(smoothingRadius - dist, 2));
}
void XPBDConstraint::solve(std::vector<Particle>& particles, float dt) {
    const int maxIterations = 10;  // 最大迭代次数
    const float tolerance = 1e-5f;  // 误差容忍度

    // 在每个时间步中，首先进行多次约束解算
    for (int iter = 0; iter < maxIterations; ++iter) {
        bool isConverged = true;

        // 每次迭代时处理所有粒子
        for (auto& pi : particles) {
            float C = computeConstraint(pi);  // 计算约束违反量
            if (std::abs(C) < tolerance) continue;  // 如果误差足够小，可以跳过该粒子

            Vec2 sumGrad2(0, 0);
            Vec2 deltaP(0, 0);

            // 计算每对粒子的约束梯度和修正量
            for (auto& pj : particles) {
                if (&pi == &pj) continue;

                Vec2 grad = computeGradient(pi, pj);
                sumGrad2 += grad.cwiseProduct(grad);  // 梯度的平方和
                deltaP += grad * (C / sumGrad2.length());  // 位置修正量
            }

            // 计算拉格朗日乘子 λ
            float lambda = C / (sumGrad2.length() + compliance);  // 计算 λ

            // 更新粒子的位置
            for (auto& pj : particles) {
                if (&pi == &pj) continue;

                Vec2 grad = computeGradient(pi, pj);
                pj.position += lambda * grad;  // 调整位置
            }
            pi.position -= lambda * deltaP;  // 更新 pi 的位置

            // 如果约束没有收敛，则标记为未收敛
            if (std::abs(C) > tolerance) {
                isConverged = false;
            }
        }
        // 如果所有粒子的约束已经满足，则提前退出
        if (isConverged) {
            break;
        }
    }

    // 处理边界条件
    applyBoundaryConditions(particles, boundary);
}