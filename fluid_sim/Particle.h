#pragma once
#include "utility.h"

struct Particle {
    Vec2 position;
    Vec2 velocity = { 0,0 };
    Vec2 force = { 0,0 };
    Vec2 prediction = { 0,0 };
    float density = 0.0f;
	float pressure = 0.0f; 
	Particle(Vec2 pos) : position(pos), velocity(0, 0), force(0, 0), prediction(0, 0), density(0.0f), pressure(0.0f) {}
};
