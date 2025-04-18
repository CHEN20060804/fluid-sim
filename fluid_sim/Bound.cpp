#include "Boundary.h"
void Boundary::ResolveCollisions(Vec2& velocity, Vec2& position, float damping )
{
    float px = position.getX();
    float py = position.getY();
	float vx = velocity.getX();
    float vy = velocity.getY();

    //вСг╫
    if (px < margin) {
        px = margin;
        if (vx < 0) vx *= -damping;
    }
    // срг╫
    else if (px> width - margin) {
        px = width - margin;
        if (vx > 0) vx *= -damping;
    }

    // обг╫
    if (py < margin) {
        py = margin;
        if (vy < 0) vy *= -damping;
    }
    // иог╫
    else if (py> height - margin) {
        py = height - margin;
        if (vy> 0) vy *= -damping;
    }
}