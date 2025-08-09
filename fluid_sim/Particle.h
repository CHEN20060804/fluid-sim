#pragma once
#include "utility.h"

struct Particle {
    Vec2 position;
    Vec2 velocity = { 0,0 };
    Vec2 force = { 0,0 };
    Vec2 prediction = { 0,0 };
    int index = 0;
    float density = 0.0f;
    float pressure = 0.0f;

    float mass = 1.0f;
    float radius = 0.0f;
	bool isSolid = false;

    __device__ __host__ Particle(Vec2 pos) : position(pos), velocity(0, 0), force(0, 0), prediction(0, 0), density(0.0f), pressure(0.0f) {}
};

