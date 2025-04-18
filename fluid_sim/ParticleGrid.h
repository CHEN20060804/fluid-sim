#pragma once
#include "utility.h"
#include "Particle.h"
#include <vector>
#include "Sort.cuh"
class ParticleGrid {
public:
	ParticleGrid(std::vector<Particle>& particles) : particles(particles) {};
	void UpdateParticleLookat();
	std::vector<Particle> ForeachPointWithinRadius(Vec2 samplePoint,float radius);
	std::pair<int, int> PositionToCellCoord(const Vec2& point);
	~ParticleGrid() {
		particles.clear();
	}

private:
	int ParticleNum;
	float cellSize = 0.1f; // �����С
	float radius;
	int HashCell(int x, int y);
	int GetKeyFromHash(int hash);
	std::vector<Particle>& particles;
	std::vector<int> startIndex{ ParticleNum };//��������������Ϊ����һ��������������ʹ��{}
	std::vector<Entry> spatiacleLookat{ ParticleNum };
	std::vector<std::pair<int, int>> offsets{
		{-1, -1}, {0, -1}, {1, -1},
		{-1, 0},          {1, 0},
		{-1, 1}, {0, 1}, {1, 1}
	};
};