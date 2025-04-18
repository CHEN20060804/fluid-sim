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
	Boundary(float width = 300, float height = 200, float margin = 50) : width(width), height(height), margin(margin) {
		position = Vec2(width / 2, height / 2);
	}
	void drawBoundary(ConsoleBuffer& console) 
	{
		for (int i = margin; i < margin + width; i++) {
			console.DrawAt(i, margin, L'#');
			console.DrawAt(i, margin + height, L'#');
		}
		for (int i = margin; i < height + margin; i++) {
			console.DrawAt(margin, i, L'#');
			console.DrawAt(width + margin, i, L'#');
		}
	}
};