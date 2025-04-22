#include "XPBD.h"
#include <numeric>
#include <execution>
#define M_PI 3.14159265358979323846
Vec2 boundaryMin(10.0f, 10.0f);
Vec2 boundaryMax(360.0f, 210.0f);
float XPBDConstraint::computeConstraint(const Particle& p) {
    return (p.density / restDensity) - 1.0f; // C = ρ/ρ₀ - 1
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
    Vec2 correction(0.0f, 0.0f); // 推回量
    Vec2 normal(0.0f, 0.0f);     // 合成法线

    // 检查 X 方向
    if (p.prediction.X() < boundaryMin.X()) {
        float penetration = boundaryMin.X() - p.prediction.X();
        correction.getX() += penetration;
        normal += Vec2(1.0f, 0.0f);  // 向右的法线
    }
    else if (p.prediction.X() > boundaryMax.X()) {
        float penetration = p.prediction.X() - boundaryMax.X();
        correction.getX() -= penetration;
        normal += Vec2(-1.0f, 0.0f); // 向左的法线
    }

    // 检查 Y 方向
    if (p.prediction.Y() < boundaryMin.Y()) {
        float penetration = boundaryMin.Y() - p.prediction.Y();
        correction.getY() += penetration;
        normal += Vec2(0.0f, 1.0f);  // 向上的法线
    }
    else if (p.prediction.Y() > boundaryMax.Y()) {
        float penetration = p.prediction.Y() - boundaryMax.Y();
        correction.getY() -= penetration;
        normal += Vec2(0.0f, -1.0f); // 向下的法线
    }

    // 如果有接触，处理反弹
    if (normal.length() > 0.0f) {
        normal = normal.normalize(); // 归一化合成法线
        float vDotN = p.velocity * normal;

        // 只处理朝向边界的速度（避免吸附）
        if (vDotN < 0.0f) {
            // 反射公式：v = v - (1 + e)(v·n)n
            p.velocity -= (1.0f + bounceDamping) * vDotN * normal;
        }

        // 推回位置
        p.prediction += correction * 1.2f;
    }
}

//void XPBDConstraint::solve(std::vector<Particle>& particles, float dt) {
//
//    std::vector<size_t> indices(particles.size());
//    std::iota(indices.begin(), indices.end(), 0);
//
//    const int maxIterations = 10;  // 最大迭代次数
//    const float tolerance = 1e-5f;  // 误差容忍度
//
//    // 第一次预测：基于当前速度进行位置预测
//    for (auto& pi : particles) {
//        pi.prediction = pi.position + pi.velocity * dt;
//    }
//
//    // 在每个时间步中，首先进行多次约束解算
//    for (int iter = 0; iter < maxIterations; ++iter) {
//        bool isConverged = true;
//
//        // 计算每个粒子的密度
//        std::for_each(std::execution::par, indices.begin(), indices.end(), [&](size_t i)
//        {
//            particles[i].density = computeDensityPrediction(particles[i], particles);
//        });
//
//        // 处理密度约束
//        for (auto& pi : particles) {
//            float C = computeConstraint(pi);  // 计算密度约束违反量
//            if (std::abs(C) < tolerance) continue;  // 如果误差足够小，可以跳过该粒子
//
//            Vec2 sumGrad2(0, 0);
//            // 计算每对粒子的约束梯度和修正量
//            for (auto& pj : particles) {
//                if (&pi == &pj) continue;
//
//                Vec2 grad = computeGradient(pi, pj);
//                sumGrad2 += grad.cwiseProduct(grad);  // 梯度的平方和
//            }
//
//            // 计算拉格朗日乘子 λ
//            float lambda = C / (sumGrad2.length() + compliance);  // 计算 λ
//
//            // 更新粒子的位置
//            for (auto& pj : particles) {
//                if (&pi == &pj) continue;
//
//                Vec2 grad = computeGradient(pi, pj);
//                pj.prediction += lambda * grad;  // 调整位置（使用预测位置进行修正）
//            }
//
//            // 如果约束没有收敛，则标记为未收敛
//            if (std::abs(C) > tolerance) {
//                isConverged = false;
//            }
//        }
//
//        // 处理表面张力约束
//        for (auto& pi : particles) {
//            for (auto& pj : particles) {
//                if (&pi == &pj) continue;
//
//                float C_surface = computeSurfaceTensionConstraint(pi, pj, smoothingRadius);
//                if (std::abs(C_surface) < tolerance) continue;  // 如果误差足够小，可以跳过该粒子
//
//                Vec2 grad_surface = computeSurfaceTensionGradient(pi, pj);
//                pi.prediction += grad_surface * C_surface * compliance;  // 修正位置
//            }
//        }
//
//        // 处理粘性约束
//        for (auto& pi : particles) {
//            for (auto& pj : particles) {
//                if (&pi == &pj) continue;
//
//                float C_viscosity = computeViscosityConstraint(pi, pj);
//                if (std::abs(C_viscosity) < tolerance) continue;  // 如果误差足够小，可以跳过该粒子
//
//                Vec2 grad_viscosity = computeViscosityGradient(pi, pj);
//                pi.prediction += grad_viscosity * C_viscosity * compliance;  // 修正位置
//            }
//        }
//
//        // 如果所有粒子的约束已经满足，则提前退出
//        if (isConverged) {
//            break;
//        }
//    }
//
//    // 第二次预测：基于约束修正后的预测位置更新实际位置
//    for (auto& pi : particles) {
//        applyBoundaryConstraint(pi, boundaryMin, boundaryMax, 0.9);
//
//        Vec2 newVelocity = (pi.prediction - pi.position) / dt;
//
//        // 限制最大速度，防止瞬时跳出
//        const float maxSpeed = 500.0f;
//        if (newVelocity.length() > maxSpeed)
//            newVelocity = newVelocity.normalize() * maxSpeed;
//
//        pi.velocity = newVelocity;
//        pi.position = pi.prediction;
//    }
//}
void XPBDConstraint::solve(std::vector<Particle>& particles, float dt) {
    std::vector<size_t> indices(particles.size());
    std::iota(indices.begin(), indices.end(), 0);

    const int maxIterations = 10;
    const float tolerance = 1e-5f;

    // 初始预测位置
    std::for_each(std::execution::par, particles.begin(), particles.end(), [&](Particle& p) {
        p.prediction = p.position + p.velocity * dt;
        });

    for (int iter = 0; iter < maxIterations; ++iter) {
        std::atomic<bool> isConverged(true);

        // 并行密度计算
        std::for_each(std::execution::par, indices.begin(), indices.end(), [&](size_t i) {
            particles[i].density = computeDensityPrediction(particles[i], particles);
         });

        // 每个粒子的预测修正累积
        std::vector<Vec2> deltaPrediction(particles.size(), Vec2(0, 0));

        // 并行密度约束处理
        std::for_each(std::execution::par, indices.begin(), indices.end(), [&](size_t i) {
            Particle& pi = particles[i];
            float C = computeConstraint(pi);
            if (std::abs(C) < tolerance) return;

            Vec2 sumGrad2(0, 0);
            std::vector<Vec2> grads(particles.size(), Vec2(0, 0));

            for (size_t j = 0; j < particles.size(); ++j) {
                if (i == j) continue;
                Vec2 grad = computeGradient(pi, particles[j]);
                grads[j] = grad;
                sumGrad2 += grad.cwiseProduct(grad);
            }

            float lambda = -C / (sumGrad2.length() + compliance / (dt * dt));

            for (size_t j = 0; j < particles.size(); ++j) {
                if (i == j) continue;
                deltaPrediction[j] += lambda * grads[j];
            }
            isConverged = false;
            });

        // 并行表面张力约束
        std::vector<Vec2> deltaSurface(particles.size(), Vec2(0, 0));
        std::for_each(std::execution::par, indices.begin(), indices.end(), [&](size_t i) {
            Particle& pi = particles[i];
            for (size_t j = 0; j < particles.size(); ++j) {
                if (i == j) continue;
                float C = computeSurfaceTensionConstraint(pi, particles[j], smoothingRadius);
                if (std::abs(C) < tolerance) continue;
                Vec2 grad = computeSurfaceTensionGradient(pi, particles[j]);
                deltaSurface[i] += grad * C * compliance;
                deltaSurface[j] -= grad * C * compliance;
            }
            });

        // 并行粘性约束
        std::vector<Vec2> deltaViscosity(particles.size(), Vec2(0, 0));
        std::for_each(std::execution::par, indices.begin(), indices.end(), [&](size_t i) {
            Particle& pi = particles[i];
            for (size_t j = 0; j < particles.size(); ++j) {
                if (i == j) continue;
                float C = computeViscosityConstraint(pi, particles[j]);
                if (std::abs(C) < tolerance) continue;
                Vec2 grad = computeViscosityGradient(pi, particles[j]);
                deltaViscosity[i] += grad * C * compliance;
				deltaViscosity[j] -= grad * C * compliance;
            }
            });

        // 统一应用位置修正
        std::for_each(std::execution::par, indices.begin(), indices.end(), [&](size_t i) {
            particles[i].prediction += deltaPrediction[i] + deltaSurface[i] + deltaViscosity[i];
            });

        if (isConverged) break;
    }

    // 并行边界约束与速度更新
    std::for_each(std::execution::par, particles.begin(), particles.end(), [&](Particle& pi) {
        applyBoundaryConstraint(pi, boundaryMin, boundaryMax, 1.0f);

        Vec2 newVelocity = (pi.prediction - pi.position) / dt;
        const float maxSpeed = 500.0f;

        if (newVelocity.length() > maxSpeed)
            newVelocity = newVelocity.normalize() * maxSpeed;

        pi.velocity = newVelocity;
        pi.position = pi.prediction;
        });
}
float XPBDConstraint::computeDensityPrediction(const Particle& pi, const std::vector<Particle>& particles) {
    float density = 0.0f;
    for (const auto& pj : particles) {
        float dist = (pi.prediction - pj.prediction).length();
        if (dist < smoothingRadius) {//poly6
            float term = smoothingRadius * smoothingRadius - dist * dist;
            density += (315.0f / (64.0f * M_PI * pow(smoothingRadius, 9))) * term * term * term;
        }
    }
    return density;
}