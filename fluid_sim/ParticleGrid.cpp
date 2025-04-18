#include "ParticleGrid.h"
#include <execution>
#include <algorithm>
#include <limits> 
#include "Sort.cuh"

const int groupHeight = 4;
const int groupWidth = 2;
const int stepIndex = 0;
const int p1 = 73856093;
const int p2 = 19349663;
const int hashTableSize = 2048;


void ParticleGrid::UpdateParticleLookat() {
	/*std::for_each(std::execution::par, particles.begin(), particles.end(), [&](const Vec2& point, size_t i)
	{
			std::pair<int, int> p = PositionToCellCoord(point);
			int cellKey = GetKeyFromHash(HashCell(p.first, p.second));
			spatiacleLookat[i] = Entry(i, cellKey);
			startIndex[i] = std::numeric_limits<int>::max();
	});*/
	std::vector<size_t> indices(particles.size());
	std::iota(indices.begin(), indices.end(), 0);  // Ìî³äÎª 0, 1, 2, ..., N-1

	std::for_each(std::execution::par, indices.begin(), indices.end(), [&](size_t i) {
		const Vec2& point = particles[i].prediction;
		std::pair<int, int> p = PositionToCellCoord(point);
		int cellKey = GetKeyFromHash(HashCell(p.first, p.second));
		spatiacleLookat[i] = Entry(i, cellKey);
		startIndex[i] = std::numeric_limits<int>::max();
		});
	Entry* spatiacleLookats;
	cudaMalloc(&spatiacleLookats, sizeof(Entry) * ParticleNum);
	cudaMemcpy(spatiacleLookats, spatiacleLookat.data(), sizeof(Entry) * ParticleNum, cudaMemcpyHostToDevice);
	int threadsPerBlock = 128;
	launchSortPairs(spatiacleLookats, ParticleNum, groupHeight, groupWidth, stepIndex,threadsPerBlock);
	
	cudaDeviceSynchronize();

	cudaMemcpy(spatiacleLookat.data(), spatiacleLookats, sizeof(Entry) * ParticleNum, cudaMemcpyDeviceToHost);
	cudaFree(spatiacleLookats);

	/*qsort(spatiacleLookat.data(), ParticleNum, sizeof(Entry), [](const void* a, const void* b) {
		return ((Entry*)a)->cellKey - ((Entry*)b)->cellKey;
		});*/

	std::for_each(std::execution::par, indices.begin(), indices.end(), [&](size_t i) {
		int key = spatiacleLookat[i].cellKey;
		int keyPrev = i == 0 ? std::numeric_limits<int>::max() : spatiacleLookat[i - 1].cellKey;
		if (key != keyPrev) {
			startIndex[key] = i;
		}
		});

}
std::vector<Particle> ParticleGrid::ForeachPointWithinRadius(Vec2 samplePoint,float radius) {

	this->radius = radius;
	std::pair<int, int> p = PositionToCellCoord(samplePoint);
	int cellKey = GetKeyFromHash(HashCell(p.first, p.second));
	std::vector<Particle> result;
	for (std::pair<int, int> i : offsets)
	{
		int key = GetKeyFromHash(HashCell(p.first + i.first, p.second + i.second));
		int cellStartIndex = startIndex[key];
		if (cellStartIndex == std::numeric_limits<int>::max()) continue;
		for (int i = cellStartIndex; i < spatiacleLookat.size(); i++)
		{
			if (spatiacleLookat[i].cellKey != key) break;
			Vec2 pos = particles[spatiacleLookat[i].index].prediction;
			Vec2 offset = samplePoint - pos;
			float sqrDist = offset.x * offset.x + offset.y * offset.y;
			if (sqrDist <= radius * radius) {
				result.push_back(particles[spatiacleLookat[i].index]);
			}
		}
	}
	return result;

}
std::pair<int, int> ParticleGrid::PositionToCellCoord(const Vec2& point)
{
	int x = static_cast<int>(std::floor(point.x / radius));
	int y = static_cast<int>(std::floor(point.y / radius));
	return { x, y };
}
int ParticleGrid::HashCell(int x, int y)
{
	return (x * p1) ^ (y * p2);
}
int ParticleGrid::GetKeyFromHash(int hash)
{
	return std::abs(hash) % hashTableSize;
}