#include "ParticleGird.h"
#include <execution>
#include <algorithm>
#include <limits>
ParticleGird::ParticleGird(std::vector<Particle> particles) : particles(particles) {
	ParticleNum = particles.size();
}
void ParticleGird::UpdateParticleLookat() {
	std::for_each(std::execution::par, particles.begin(), particles.end(), [&](const Vec2& point, size_t i)
	{
			std::pair<int, int> p = PositionToCellCoord(point);
			int cellKey = GetKeyFromHash(HashCell(p.first, p.second));
			spatiacleLookat[i] = Entry(i, cellKey);
			startIndex[i] = std::numeric_limits<int>::max();
	});
	qsort(spatiacleLookat.data(), ParticleNum, sizeof(Entry), [](const void* a, const void* b) {
		return ((Entry*)a)->cellKey - ((Entry*)b)->cellKey;
		});
	std::for_each(std::execution::par, particles.begin(), particles.end(), [&](const Vec2& point, size_t i)
		{
			int key = spatiacleLookat[i].cellKey;
			int keyPrev = i == 0 ? std::numeric_limits<int>::max() : spatiacleLookat[i - 1].cellKey;
			if (key != keyPrev) {
				startIndex[key] = i;
			}
		});

}
std::vector<Particle> ParticleGird::ForeachPointWithinRadius(Vec2 samplePoint) {
	std::pair<int, int> p = PositionToCellCoord(samplePoint);
	int cellKey = GetKeyFromHash(HashCell(p.first, p.second));
	std::vector<Particle> result;
	for (std::pair<int, int> i : offsets)
	{
		int key = GetKeyFromHash(HashCell(p.first + i.first, p.second + i.second));
		int cellStartIndex = startIndex[key];
		for (int i = cellStartIndex; i < spatiacleLookat.size(); i++)
		{
			if (spatiacleLookat[i].cellKey != key) break;
			result.push_back(particles[spatiacleLookat[i].index]);
		}
	}
	return result;

}
std::pair<int, int> ParticleGird::PositionToCellCoord(const Vec2& point)
{

}
int HashCell(int x, int y)
{

}
int GetKeyFromHash(int hash)
{

}