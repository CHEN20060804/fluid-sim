#pragma once
#include "utility.h"
#include "Particle.h"
#include <vector>

struct Entry {
	int index;
	int cellKey;
	Entry(int idx, int key) : index(idx), cellKey(key) {};
};
class ParticleGird {
public:
	ParticleGird(std::vector<Particle> particles) {};
	void UpdateParticleLookat();
	std::vector<Particle> ForeachPointWithinRadius(Vec2 samplePoint);
	std::pair<int, int> PositionToCellCoord(const Vec2& point);
private:
	int ParticleNum;
	std::vector<Particle> particles;
	std::vector<int> startIndex{ ParticleNum };//编译器会优先认为他是一个函数，所以请使用{}
	std::vector<Entry> spatiacleLookat{ ParticleNum };
	std::vector<std::pair<int, int>> offsets{
		{-1, -1}, {0, -1}, {1, -1},
		{-1, 0},          {1, 0},
		{-1, 1}, {0, 1}, {1, 1}
	};
};