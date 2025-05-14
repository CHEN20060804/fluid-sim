#pragma once

#define WIDTH 800.0f
#define HEIGHT 450.0f
#include "utility.h"
#include "Particle.h"
#include "ConsoleBuffer.h"
class Boundary {
private:
	float width;
	float height;
	float margin;
public:
	void ResolveCollisions(Particle& p, float damping = 0.4f);
	Boundary(float width = WIDTH, float height = HEIGHT, float margin = 1.0f) : width(width), height(height), margin(margin) {
	}
	void drawBoundary(ConsoleBuffer& console);
};