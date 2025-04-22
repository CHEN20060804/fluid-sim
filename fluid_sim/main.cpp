#include <chrono>
#include <thread>
#include "ConsoleBuffer.h"
#include "Boundary.h"
#include <locale>
#include <iostream>
#include "Particle.h"
#include "Integrator.h"
#include "SPH.h"
#include <fstream>
#include "XPBD.h"
#include <random>
const int gravity = 20;
#include <random>


void generateRandomParticles(std::vector<Particle>& particles, int count, float xMin, float xMax, float yMin, float yMax) {
    std::random_device rd;
    std::mt19937 gen(rd()); // 使用随机种子初始化
    std::uniform_real_distribution<float> distX(xMin, xMax);
    std::uniform_real_distribution<float> distY(yMin, yMax);

    for (int i = 0; i < count; ++i) {
        float x = distX(gen);
        float y = distY(gen);
        particles.emplace_back(Particle(Vec2(x, y)));
    }
}



int main() {
    std::wcout.imbue(std::locale(""));
    ConsoleBuffer consoleBuffer;
    Boundary boundary;
    std::vector<Particle> particles;
	generateRandomParticles(particles, 800, 50, 350, 50, 350);
    SPHSolver sphSolver(particles);
	
	ParticleGrid::getInstance().init(particles);
	
	//XPBDConstraint xpbdConstraint;
	//xpbdConstraint.smoothingRadius = 8.0f;
	//xpbdConstraint.restDensity = 500.0f;
	//xpbdConstraint.compliance = 0.0001f;


    while (true) {
        consoleBuffer.Clear();
        boundary.drawBoundary(consoleBuffer);
		sphSolver.simulateStep(0.1f);
		//xpbdConstraint.solve(particles, 0.01f);
		for (int i = 0; i < particles.size(); i++) {
            int x = static_cast<int>(std::round(particles[i].position.getX()));
            int y = static_cast<int>(std::round(particles[i].position.getY()));
            consoleBuffer.DrawAt(x, y, L'●');
		}
        consoleBuffer.Flush();
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    std::cin.get();
}
