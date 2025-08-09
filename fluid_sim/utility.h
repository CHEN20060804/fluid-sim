#pragma once
#include <cmath> // for sqrtf
#ifndef __CUDA_ARCH__
#include <iostream>
#endif
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
__host__ __device__ struct Entry {
    int index;
    int cellKey;
    __host__ __device__ Entry() : index(0), cellKey(0) {}
    __host__ __device__ Entry(int idx, int key) : index(idx), cellKey(key) {}

    __host__ __device__ bool operator<(const Entry& other) const {
        return cellKey < other.cellKey;
    }
};
struct Vec2 {

    float x, y;

    __host__ __device__ Vec2(float x, float y) : x(x), y(y) {}
    __host__ __device__ Vec2() : x(0.0f), y(0.0f) {}

    __host__ __device__ float& getX() { return x; }
    __host__ __device__ float& getY() { return y; }
    __host__ __device__ float X() const { return x; }
    __host__ __device__ float Y() const { return y; }

    __host__ __device__ Vec2 operator+(const Vec2& b) const { return Vec2(x + b.x, y + b.y); }
    __host__ __device__ Vec2 operator-(const Vec2& b) const { return Vec2(x - b.x, y - b.y); }
    __host__ __device__ Vec2 operator-() const { return Vec2(-x, -y); }
    __host__ __device__ Vec2 operator/(float s) const { return Vec2(x / s, y / s); }
    __host__ __device__ Vec2 operator*(float s) const { return Vec2(x * s, y * s); }

    __host__ __device__ float operator*(const Vec2& b) const { return x * b.x + y * b.y; }

    __host__ __device__ Vec2& operator+=(const Vec2& b) {
        x += b.x; y += b.y; return *this;
    }
    __host__ __device__ Vec2& operator-=(const Vec2& b) {
        x -= b.x; y -= b.y; return *this;
    }
    __host__ __device__ Vec2& operator*=(float s) {
        x *= s; y *= s; return *this;
    }
    __host__ __device__ Vec2& operator/=(float s) {
        x /= s; y /= s; return *this;
    }

    __host__ __device__ bool operator==(const Vec2& v) const {
        return x == v.x && y == v.y;
    }

    __host__ __device__ float length() const {
        return sqrtf(x * x + y * y);
    }

    __host__ __device__ Vec2 normalize() const {
        float len = length();
        return len > 0.0f ? *this / len : Vec2(0.0f, 0.0f);
    }

    __host__ __device__ Vec2 cwiseProduct(const Vec2& v) const {
        return Vec2(x * v.x, y * v.y);
    }

#ifndef __CUDA_ARCH__
    friend std::ostream& operator<<(std::ostream& os, const Vec2& vec) {
        os << "(" << vec.x << ", " << vec.y << ")";
        return os;
    }
#endif
};

__host__ __device__ inline Vec2 operator*(float s, Vec2 v) {
    return v * s;
}
