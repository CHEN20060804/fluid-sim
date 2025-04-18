#include "Boundary.h"
void Boundary::ResolveCollisions(Vec2& velocity, Vec2& position, float damping )
{
    float px = position.getX();
    float py = position.getY();
	float vx = velocity.getX();
    float vy = velocity.getY();

    //��ǽ
    if (px < margin) {
        px = margin;
        if (vx < 0) vx *= -damping;
    }
    // ��ǽ
    else if (px> width - margin) {
        px = width - margin;
        if (vx > 0) vx *= -damping;
    }

    // ��ǽ
    if (py < margin) {
        py = margin;
        if (vy < 0) vy *= -damping;
    }
    // ��ǽ
    else if (py> height - margin) {
        py = height - margin;
        if (vy> 0) vy *= -damping;
    }
}