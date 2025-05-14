#pragma once
#include "PBF.h"
#include <numeric>
#include <execution>
//#include "ParticleGrid.h"
#define M_PI 3.14159265358979323846

float PBFConstraint::computeConstraint(const Particle& p) {
    return (p.density / restDensity) - 1.0f; // C = ρ/ρ₀ - 1
}
Vec2 PBFConstraint::computeGradient(const Particle& pi, const Particle& pj) {
    Vec2 r = pi.prediction - pj.prediction;
    float dist = r.length();
    if (dist <= 0.00001f || dist > smoothingRadius) return Vec2();//距离过小视为重合
    return r.normalize() * (-45.0f / (M_PI * pow(smoothingRadius, 6))) * (pow(smoothingRadius - dist, 2));
}

float PBFConstraint::computeSurfaceTensionConstraint(const Particle& pi, const Particle& pj, float restDistance) {
    return (pi.prediction - pj.prediction).length() - restDistance;
}
Vec2 PBFConstraint::computeSurfaceTensionGradient(const Particle& pi, const Particle& pj)
{
    Vec2 r = pi.prediction - pj.prediction;
    float len = r.length();
    if (len <= 0.00001f || len > smoothingRadius) return Vec2();
    return r / len;
}

float PBFConstraint::computeViscosityConstraint(const Particle& pi, const Particle& pj) {
    Vec2 vij = pi.velocity - pj.velocity;
    Vec2 dx = pi.prediction - pj.prediction;
    return dx * vij;
}
Vec2 PBFConstraint::computeViscosityGradient(const Particle& pi, const Particle& pj)
{
    Vec2 vij = pi.velocity - pj.velocity;
    float len = vij.length();
    if (len <= 0.00001f || len > smoothingRadius) return Vec2();//距离过小视为重合
    return vij;
}
void PBFConstraint::applyBoundaryConstraint(Particle& p, const Vec2& boundaryMin, const Vec2& boundaryMax, float bounceDamping) {
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
//void PBFConstraint::solve(std::vector<Particle>& particles, float dt) {
//    std::vector<size_t> indices(particles.size());
//    std::iota(indices.begin(), indices.end(), 0);
//
//    const int maxIterations = 5;
//    const float tolerance = 1e-5f;
//
//    // 初始预测位置
//    std::for_each(std::execution::par, particles.begin(), particles.end(), [&](Particle& p) {
//        p.prediction = p.position + p.velocity * dt;
//        });
//    ParticleGrid::getInstance().UpdateParticleLookat();
//
//    for (int iter = 0; iter < maxIterations; ++iter) {
//        std::atomic<bool> isConverged(true);
//
//        // 并行密度计算
//        std::for_each(std::execution::par, indices.begin(), indices.end(), [&](size_t i) {
//            std::vector<Particle>& other = ParticleGrid::getInstance().ForeachPointWithinRadius(particles[i].prediction, this->smoothingRadius);
//            particles[i].density = computeDensityPrediction(particles[i], other);
//            });
//
//        // 每个粒子的预测修正累积
//        std::vector<Vec2> deltaPrediction(particles.size(), Vec2(0, 0));
//
//        // 并行密度约束处理
//        std::for_each(std::execution::par, indices.begin(), indices.end(), [&](size_t i) {
//            std::vector<Particle>& other = ParticleGrid::getInstance().ForeachPointWithinRadius(particles[i].prediction, this->smoothingRadius);
//            Particle& pi = particles[i];
//            float C = computeConstraint(pi);
//            if (std::abs(C) < tolerance) return;
//
//            Vec2 sumGrad2(0, 0);
//            std::vector<Vec2> grads(other.size(), Vec2(0, 0));
//
//            for (size_t j = 0; j < other.size(); ++j) {
//                if (i == other[j].index) continue;
//                Vec2 grad = computeGradient(pi, other[j]);
//                grads[j] = grad;
//                sumGrad2 += grad.cwiseProduct(grad);
//            }
//
//            float lambda = -C / (sumGrad2.length() + epsilon);
//
//            for (size_t j = 0; j < other.size(); ++j) {
//                if (i == j) continue;
//                deltaPrediction[other[j].index] += lambda * grads[j];
//            }
//            isConverged = false;
//            });
//
//        // 并行表面张力约束
//        std::vector<Vec2> deltaSurface(particles.size(), Vec2(0, 0));
//        std::for_each(std::execution::par, indices.begin(), indices.end(), [&](size_t i) {
//            std::vector<Particle>& other = ParticleGrid::getInstance().ForeachPointWithinRadius(particles[i].prediction, this->smoothingRadius);
//            Particle& pi = particles[i];
//            for (size_t j = 0; j < other.size(); ++j) {
//                if (i == j) continue;
//                float C = computeSurfaceTensionConstraint(pi, other[j], smoothingRadius);
//                if (std::abs(C) < tolerance) continue;
//                Vec2 grad = computeSurfaceTensionGradient(pi, other[j]);
//                deltaSurface[i] += grad * C;
//                deltaSurface[other[j].index] -= grad * C;
//            }
//            });
//
//        // 并行粘性约束
//        std::vector<Vec2> deltaViscosity(particles.size(), Vec2(0, 0));
//        std::for_each(std::execution::par, indices.begin(), indices.end(), [&](size_t i) {
//            std::vector<Particle>& other = ParticleGrid::getInstance().ForeachPointWithinRadius(particles[i].prediction, this->smoothingRadius);
//            Particle& pi = particles[i];
//            for (size_t j = 0; j < other.size(); ++j) {
//                if (i == j) continue;
//                float C = computeViscosityConstraint(pi, other[j]);
//                if (std::abs(C) < tolerance) continue;
//                Vec2 grad = computeViscosityGradient(pi, other[j]);
//                deltaViscosity[i] += grad * C;
//                deltaViscosity[other[j].index] -= grad * C;
//            }
//            });
//
//        // 统一应用位置修正
//        const float maxDelta = 0.1f;
//        std::for_each(std::execution::par, indices.begin(), indices.end(), [&](size_t i) {
//            particles[i].prediction += std::clamp(deltaPrediction[i].length(), 0.0f, maxDelta) * deltaPrediction[i].normalize() +
//                std::clamp(deltaSurface[i].length(), 0.0f, maxDelta) * deltaSurface[i].normalize() +
//                std::clamp(deltaViscosity[i].length(), 0.0f, maxDelta) * deltaViscosity[i].normalize();
//            });
//
//        if (isConverged) break;
//    }
//
//    // 并行边界约束与速度更新
//    std::for_each(std::execution::par, particles.begin(), particles.end(), [&](Particle& pi) {
//        applyBoundaryConstraint(pi, boundaryMin1, boundaryMax1, 1.0f);
//
//        Vec2 newVelocity = (pi.prediction - pi.position) / dt;
//        const float maxSpeed = 500.0f;
//
//        if (newVelocity.length() > maxSpeed)
//            newVelocity = newVelocity.normalize() * maxSpeed;
//
//        pi.velocity = newVelocity;
//        pi.position = pi.prediction;
//        });
//}

void PBFConstraint::solve(std::vector<Particle>& particles, float dt) {
    std::vector<size_t> indices(particles.size());
    std::iota(indices.begin(), indices.end(), 0);

    const int maxIterations = 5;
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

            float lambda = -C / (sumGrad2.length() + epsilon);

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
                deltaSurface[i] += grad * C;
                deltaSurface[j] -= grad * C;
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
                deltaViscosity[i] += grad * C;
                deltaViscosity[j] -= grad * C;
            }
            });

        // 统一应用位置修正
        const float maxDelta = 0.1f;
        std::for_each(std::execution::par, indices.begin(), indices.end(), [&](size_t i) {
            particles[i].prediction += std::clamp(deltaPrediction[i].length(), 0.0f, maxDelta) * deltaPrediction[i].normalize() +
                std::clamp(deltaSurface[i].length(), 0.0f, maxDelta) * deltaSurface[i].normalize() +
                std::clamp(deltaViscosity[i].length(), 0.0f, maxDelta) * deltaViscosity[i].normalize();
            });

        if (isConverged) break;
    }

    // 并行边界约束与速度更新
    std::for_each(std::execution::par, particles.begin(), particles.end(), [&](Particle& pi) {
        applyBoundaryConstraint(pi, boundaryMin1, boundaryMax1, 1.0f);

        Vec2 newVelocity = (pi.prediction - pi.position) / dt;
        const float maxSpeed = 500.0f;

        if (newVelocity.length() > maxSpeed)
            newVelocity = newVelocity.normalize() * maxSpeed;

        pi.velocity = newVelocity;
        pi.position = pi.prediction;
        });
}
float PBFConstraint::computeDensityPrediction(const Particle& pi, const std::vector<Particle>& particles) {
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