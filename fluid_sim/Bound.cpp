#include "Boundary.h"
void Boundary::ResolveCollisions(Particle& p, float damping)
{
    float px = p.position.getX();
    float py = p.position.getY();
    float vx = p.velocity.getX();
    float vy = p.velocity.getY();
    float r = p.radius;
    // 碰撞标志
    bool collided = false;

    // 左右边界
    if (px - r < margin + 2) {
        px = margin + r + 2;
        vx = -vx * damping;
        collided = true;
    }
    else if (px + r > width - margin) {
        px = width - margin - r;
        vx = -vx * damping;
        collided = true;
    }

    // 上下边界
    if (py - r < margin + 1) {
        py = margin + r + 1;
        vy = -vy * damping;
        collided = true;
    }
    else if (py + r > height - margin) {
        py = height - margin - r;
        vy = -vy * damping;
        collided = true;
    }

    // 可选：碰撞后给个小阻尼修正，避免剧烈反复震荡
    if (collided) {
        vx *= 0.95f;
        vy *= 0.95f;
    }

    p.position = Vec2(px, py);
    p.velocity = Vec2(vx, vy);
}

void Boundary::drawBoundary(ConsoleBuffer& console)
{
    for (int i = 1; i < 3 + width; i++) {
        console.DrawAt(i, 1, L'━', FOREGROUND_BLUE | FOREGROUND_GREEN | FOREGROUND_RED);
        console.DrawAt(i, height + 5, L'━', FOREGROUND_BLUE | FOREGROUND_GREEN | FOREGROUND_RED);
    }
    console.DrawAt(1, 1, L'┏');
    console.DrawAt(width + 3, 1, L'┓', FOREGROUND_BLUE | FOREGROUND_GREEN | FOREGROUND_RED);
    console.DrawAt(1, height + 5, L'┗', FOREGROUND_BLUE | FOREGROUND_GREEN | FOREGROUND_RED);
    console.DrawAt(width + 3, height + 5, L'┛', FOREGROUND_BLUE | FOREGROUND_GREEN | FOREGROUND_RED);
    for (int i = 1; i < height + 5; i++) {
        console.DrawAt(1, i, L'┃', FOREGROUND_BLUE | FOREGROUND_GREEN | FOREGROUND_RED);
        console.DrawAt(width + 3, i, L'┃', FOREGROUND_BLUE | FOREGROUND_GREEN | FOREGROUND_RED);
    }
}