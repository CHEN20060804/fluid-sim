#pragma once  
#include "utility.h"  
#include "Particle.h"  
#include <vector>  
#include <algorithm>  
#include <limits>  
#undef max

const int hashTableSize = 2048;

class ParticleGrid {
public:
    
    // 获取唯一实例  
    static ParticleGrid& getInstance() {
        static ParticleGrid instance;
        return instance;
    }
    int HashCell(int x, int y);
    std::vector<int> startIndex;
    std::vector<Entry> spatiacleLookat;
    // 初始化方法（只能调用一次）  
    void init(std::vector<Particle>& particlesRef) {
        particles = &particlesRef;
        ParticleNum = particlesRef.size();
        startIndex.resize(hashTableSize, std::numeric_limits<int>::max());
        spatiacleLookat.resize(ParticleNum);
    }
    void addParticle(const Particle& newParticle);
	std::vector<Particle>& getParticles() {
		return *particles;
	}

    // 粒子相关接口  
    void UpdateParticleLookat();
    std::vector<std::reference_wrapper<Particle>> ForeachPointWithinRadius(Vec2 samplePoint, float radius);
    std::vector<std::reference_wrapper<Particle>> ForeachMouseWithinRadius(Vec2 samplePoint, float radius);
    std::pair<int, int> PositionToCellCoord(const Vec2& point);

    // 禁止拷贝与赋值  
    ParticleGrid(const ParticleGrid&) = delete;
    ParticleGrid& operator=(const ParticleGrid&) = delete;

private:
    // 私有构造函数  
    ParticleGrid() = default;
    ~ParticleGrid() = default;

    int ParticleNum = 0;
    float radius = 20.0f;//3.0

    
    int GetKeyFromHash(int hash);

    std::vector<Particle>* particles = nullptr;
    

    std::vector<std::pair<int, int>> offsets{
        {-1, -1}, {0, -1}, {1, -1},
        {-1, 0},          {1, 0},
        {-1, 1}, {0, 1}, {1, 1}
    };
};