#pragma once
#include <iostream>
class Vec2 {
private:
    float x, y;

public:
    Vec2(float x, float y);
    Vec2();
    float getX() const;
    float getY() const;
    void setX(float x);
    void setY(float y);
    Vec2 operator+(const Vec2& b) const;
    Vec2 operator-(const Vec2& b) const;
    Vec2 operator-();
    Vec2 operator/(float s) const;
    Vec2 operator*(float s) const;
    Vec2& operator+=(const Vec2& b);
    Vec2& operator-=(const Vec2& b);
	Vec2& operator*=(float s);
	Vec2& operator/=(float s);
	float length() const;
	Vec2 normalize() const;
	Vec2 cwiseProduct(const Vec2& v) const;

    friend std::ostream& operator<<(std::ostream& os, const Vec2& vec);
};

Vec2 operator*(Vec2 v);
