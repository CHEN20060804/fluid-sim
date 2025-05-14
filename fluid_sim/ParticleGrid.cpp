#include "ParticleGrid.h"
#include <execution>
#include <algorithm>
#include <limits> 
#include<fstream>

const int groupHeight = 4;
const int groupWidth = 2;
const int stepIndex = 0;
const int p1 = 73856093;
const int p2 = 19349663;


void ParticleGrid::UpdateParticleLookat() {
	std::vector<size_t> indices(ParticleNum);
	std::iota(indices.begin(), indices.end(), 0);

	//初始化spatiacleLookat
	std::for_each(std::execution::par, indices.begin(), indices.end(), [&](size_t i) {
		const Vec2& point = particles->at(i).prediction;
		std::pair<int, int> cell = PositionToCellCoord(point);
		int hash = HashCell(cell.first, cell.second);
		int key = GetKeyFromHash(hash);

		spatiacleLookat[i] = Entry(i, key);
		});

	std::sort(std::execution::par, spatiacleLookat.begin(), spatiacleLookat.end(),
		[](const Entry& a, const Entry& b) {
			return a.cellKey < b.cellKey;
		});

	//初始化startIndex
	for (size_t i = 0; i < spatiacleLookat.size(); ++i) {
		int key = spatiacleLookat[i].cellKey;
		if (i == 0 || key != spatiacleLookat[i - 1].cellKey) {
			if (key >= 0 && key < static_cast<int>(startIndex.size()))
				startIndex[key] = static_cast<int>(i);
		}
	}
}
std::vector<std::reference_wrapper<Particle>> ParticleGrid::ForeachPointWithinRadius(Vec2 samplePoint, float radius) {
	this->radius = radius;
	std::pair<int, int> p = PositionToCellCoord(samplePoint);
	int cellKey = GetKeyFromHash(HashCell(p.first, p.second));
	std::vector<std::reference_wrapper<Particle>> result;

	for (const std::pair<int, int>& offset : offsets) {
		int key = GetKeyFromHash(HashCell(p.first + offset.first, p.second + offset.second));

		int cellStartIndex = startIndex[key];
		if (cellStartIndex == std::numeric_limits<int>::max()) continue;

		for (int i = cellStartIndex; i < spatiacleLookat.size(); i++) {
			if (spatiacleLookat[i].cellKey != key) break;

			Vec2 pos = particles->at(spatiacleLookat[i].index).prediction;
			Vec2 offset = samplePoint - pos;
			float sqrDist = offset.getX() * offset.getX() + offset.getY() * offset.getY();

			if (sqrDist <= radius * radius) {

				result.push_back(std::ref(particles->at(spatiacleLookat[i].index)));
			}
		}
	}
	return result;
}

std::pair<int, int> ParticleGrid::PositionToCellCoord(const Vec2& point)
{
	int x = static_cast<int>(std::floor(point.X() / radius));
	int y = static_cast<int>(std::floor(point.Y() / radius));
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

void ParticleGrid::addParticle(const Particle& newParticle) {
	if (!particles) return;

	//添加粒子
	particles->push_back(newParticle);
	int newIndex = particles->size() - 1;

	//更新粒子数据
	Vec2 pos = newParticle.position;
	auto [x, y] = PositionToCellCoord(pos);
	int hash = HashCell(x, y);
	if (spatiacleLookat.size() <= newIndex) {
		spatiacleLookat.resize(newIndex + 1);
	}
	Entry newEntry{ newIndex, hash };
	spatiacleLookat[newIndex] = newEntry;

	//更新内存
	if (startIndex.size() <= particles->size()) {
		startIndex.resize(particles->size() + 1, std::numeric_limits<int>::max());
	}
	int key = GetKeyFromHash(hash);
	if (key >= 0 && key < startIndex.size()) {
		if (startIndex[key] == std::numeric_limits<int>::max()) {
			startIndex[key] = newIndex;
		}
	}
	std::sort(spatiacleLookat.begin(), spatiacleLookat.end(), [](const Entry& a, const Entry& b) {
		return a.cellKey < b.cellKey;
		});
	

	ParticleNum = particles->size();
}
std::vector<std::reference_wrapper<Particle>> ParticleGrid::ForeachMouseWithinRadius(Vec2 samplePoint, float radius) {
	std::pair<int, int> centerCell = PositionToCellCoord(samplePoint);
	std::vector<std::reference_wrapper<Particle>> result;

	// 计算覆盖范围
	int gridRangeX = static_cast<int>(std::ceil(radius / groupWidth));
	int gridRangeY = static_cast<int>(std::ceil(radius / groupHeight));

	for (int dx = -gridRangeX; dx <= gridRangeX; ++dx) {
		for (int dy = -gridRangeY; dy <= gridRangeY; ++dy) {
			int cellX = centerCell.first + dx;
			int cellY = centerCell.second + dy;

			int key = GetKeyFromHash(HashCell(cellX, cellY));

			int cellStartIndex = startIndex[key];
			if (cellStartIndex == std::numeric_limits<int>::max()) continue;

			for (int i = cellStartIndex; i < spatiacleLookat.size(); i++) {
				if (spatiacleLookat[i].cellKey != key) break;

				Vec2 pos = particles->at(spatiacleLookat[i].index).prediction;
				Vec2 offset = samplePoint - pos;
				float sqrDist = offset.getX() * offset.getX() + offset.getY() * offset.getY();

				// 判断是否在范围内
				if (sqrDist <= radius * radius) {
					result.push_back(std::ref(particles->at(spatiacleLookat[i].index)));
				}
			}
		}
	}
	return result;
}