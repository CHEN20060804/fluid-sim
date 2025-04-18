#pragma once
#include "utility.h"
#include "Particle.h"
#include <vector>
#include "Sort.cuh"
class ParticleGrid {
public:
	ParticleGrid(std::vector<Particle>& particles) : particles(particles) {
		ParticleNum = particles.size();
	};
	void UpdateParticleLookat();
	std::vector<Particle> ForeachPointWithinRadius(Vec2 samplePoint, float radius);
	std::pair<int, int> PositionToCellCoord(const Vec2& point);
	~ParticleGrid() {
		particles.clear();
	}

private:
	int ParticleNum;
    float radius = 0.1f;
	int HashCell(int x, int y);
	int GetKeyFromHash(int hash);
	std::vector<Particle>& particles;
	std::vector<int> startIndex = std::vector<int>(ParticleNum);//编译器会优先认为他是一个函数，所以请使用{}
	std::vector<Entry> spatiacleLookat = std::vector<Entry>(ParticleNum);
	std::vector<std::pair<int, int>> offsets{
		{-1, -1}, {0, -1}, {1, -1},
		{-1, 0},          {1, 0},
		{-1, 1}, {0, 1}, {1, 1}
	};
};