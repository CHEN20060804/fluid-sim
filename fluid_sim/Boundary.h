#ifndef DEBUG
#define DEBUG


#include "utility.h"
#include "ConsoleBuffer.h"
class Boundary {
private:
	Vec2 position;
	float width;
	float height;
	float margin;
public:
	void ResolveCollisions(Vec2& velocity, Vec2& position, float damping = 0.5f);
	Boundary(float width = 350, float height = 200, float margin = 10) : width(width), height(height), margin(margin) {
		position = Vec2(width / 2, height / 2);
	}
	void drawBoundary(ConsoleBuffer& console);
}; 
#endif // DEBUG