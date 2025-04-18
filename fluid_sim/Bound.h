#include "utility.h"
class Bound {
private:
	Vec2 position;
	float width;
	float height;
	float margin;
public:
	void ResolveCollisions(Vec2& velocity, Vec2& position, float damping = 0.5f);
	Bound(float width, float height, float margin) : width(width), height(height), margin(margin) {
		position = Vec2(width / 2, height / 2);
	}
  
};