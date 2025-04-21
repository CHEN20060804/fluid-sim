#pragma once  
#include "utility.h"  
#include "Particle.h"  
#include <vector>  
#include "Sort.cuh"  
#include <algorithm>  
#include <limits>  

// �޸�Ǳ�ڵĺ��ͻ����  
#ifdef max  
#undef max  
#endif  

const int hashTableSize = 2048;  

class ParticleGrid {  
public:  
   // ��ȡΨһʵ��  
   static ParticleGrid& getInstance() {  
       static ParticleGrid instance;  
       return instance;  
   }  

   // ��ʼ��������ֻ�ܵ���һ�Σ�  
   void init(std::vector<Particle>& particlesRef) {  
       particles = &particlesRef;  
       ParticleNum = particlesRef.size();  
       startIndex.resize(hashTableSize, std::numeric_limits<int>::max());  
       spatiacleLookat.resize(ParticleNum);  
   }  

   // ������ؽӿ�  
   void UpdateParticleLookat();  
   std::vector<Particle> ForeachPointWithinRadius(Vec2 samplePoint, float radius);  
   std::pair<int, int> PositionToCellCoord(const Vec2& point);  

   // ��ֹ�����븳ֵ  
   ParticleGrid(const ParticleGrid&) = delete;  
   ParticleGrid& operator=(const ParticleGrid&) = delete;  

private:  
   // ˽�й��캯��  
   ParticleGrid() = default;  
   ~ParticleGrid() = default;  

   int ParticleNum = 0;  
   float radius = 3.0f;  

   int HashCell(int x, int y);  
   int GetKeyFromHash(int hash);  

   std::vector<Particle>* particles = nullptr;  
   std::vector<int> startIndex;  
   std::vector<Entry> spatiacleLookat;  

   std::vector<std::pair<int, int>> offsets{  
       {-1, -1}, {0, -1}, {1, -1},  
       {-1, 0},          {1, 0},  
       {-1, 1}, {0, 1}, {1, 1}  
   };  
};