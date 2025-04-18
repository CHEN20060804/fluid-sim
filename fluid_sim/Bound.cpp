#include "Bound.h"
void Bound::ResolveCollisions(Vec2& velocity, Vec2& position, float damping = 0.5f)
{
    if (position.x < margin) {
        position.x = margin;
        if (velocity.x < 0) velocity.x *= -damping;
    }
    // срг╫
    else if (position.x > width - margin) {
        position.x = width - margin;
        if (velocity.x > 0) velocity.x *= -damping;
    }

    // обг╫
    if (position.y < margin) {
        position.y = margin;
        if (velocity.y < 0) velocity.y *= -damping;
    }
    // иог╫
    else if (position.y > height - margin) {
        position.y = height - margin;
        if (velocity.y > 0) velocity.y *= -damping;
    }
}