#include "Boundary.h"
void Boundary::ResolveCollisions(Vec2& velocity, Vec2& position, float damping )
{
    float px = position.getX();
    float py = position.getY();
	float vx = velocity.getX();
    float vy = velocity.getY();

    //×óÇ½
    if (px < margin) {
        px = margin;
        if (vx < 0) vx *= -damping;
    }
    // ÓÒÇ½
    else if (px> width - margin) {
        px = width - margin;
        if (vx > 0) vx *= -damping;
    }

    // ÏÂÇ½
    if (py < margin) {
        py = margin;
        if (vy < 0) vy *= -damping;
    }
    // ÉÏÇ½
    else if (py> height - margin) {
        py = height - margin;
        if (vy> 0) vy *= -damping;
    }
}

void Boundary::drawBoundary(ConsoleBuffer& console)
{
    for (int i = margin; i < margin + width; i++) {
        console.DrawAt(i, margin, L'©¥');
        console.DrawAt(i, margin + height, L'©¥');
    }
    console.DrawAt(margin, margin, L'©³');
    console.DrawAt(width + margin, margin, L'©·');
    console.DrawAt(margin, margin + height, L'©»');
    console.DrawAt(width + margin, margin + height, L'©¿');
    for (int i = margin + 1; i < height + margin; i++) {
        console.DrawAt(margin, i, L'©§');
        console.DrawAt(width + margin, i, L'©§');
    }
}