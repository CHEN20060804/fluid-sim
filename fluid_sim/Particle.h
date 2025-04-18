#pragma once
#include "utility.h"

struct Particle {
    Vec2 position;
    Vec2 velocity;
    Vec2 force;
    Vec2 prediction;
    float density = 0.0f;
	float pressure = 0.0f;
    float lambda = 0.0f; 
};
