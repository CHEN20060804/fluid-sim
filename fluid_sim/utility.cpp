#include "utility.h"
#include <cmath>  

//构造函数
Vec2::Vec2(float x, float y) : x(x), y(y) {}
Vec2::Vec2() : x(0.0f), y(0.0f) {} // 默认构造函数，初始化为(0, 0)


float& Vec2::getX()  {
    return x;
}
float& Vec2::getY()  {
    return y;
}

// 加法运算符重载
Vec2 Vec2::operator+(const Vec2& b) const {
    return Vec2(x + b.x, y + b.y);
}

// 减法运算符重载
Vec2 Vec2::operator-(const Vec2& b) const {
    return Vec2(x - b.x, y - b.y);
}
// 负号运算符重载
Vec2 Vec2::operator-(){
	return Vec2(-x, -y);
}


// 除法运算符重载
Vec2 Vec2::operator/(float s) const {
    if (s != 0) {
        return Vec2(x / s, y / s);
    }
    // 处理除以0的情况，可以抛出异常或返回一个默认的Vec2
    return Vec2(0.0f, 0.0f);
}

// 乘法运算符重载
Vec2 Vec2::operator*(float s) const {
    return Vec2(x * s, y * s);
}

// 加法赋值运算符重载
Vec2& Vec2::operator+=(const Vec2& b) {
    x += b.x;
    y += b.y;
    return *this;
}

// 减法赋值运算符重载
Vec2& Vec2::operator-=(const Vec2& b) {
    x -= b.x;
    y -= b.y;
    return *this;
}

// 乘法赋值运算符重载
Vec2& Vec2::operator*=(float s) {
    x *= s;
    y *= s;
    return *this;
}

// 除法赋值运算符重载
Vec2& Vec2::operator/=(float s) {
    if (s != 0) {
        x /= s;
        y /= s;
    }
    else {
        // 可以考虑抛出异常或者做错误处理
        x = y = 0.0f;
    }
    return *this;
}

// 获取向量的长度
float Vec2::length() const {
    return std::sqrt(x * x + y * y);
}

// 归一化向量方法
Vec2 Vec2::normalize() const {
   float len = length();
   if (len > 0) {
       return Vec2(x / len, y / len); // 返回归一化后的新向量
   }
   // 如果长度为0，避免除零错误，可以返回一个默认的Vec2
   return Vec2(0.0f, 0.0f);
}

//重载<<运算符
std::ostream& operator<<(std::ostream& os, const Vec2& vec) {
	os << "(" << vec.x << ", " << vec.y << ")";
	return os;
}

//重载数乘运算符
Vec2 operator*(float s, Vec2 v) {
    return v.operator*(s);
}

}

Vec2 Vec2::cwiseProduct(const Vec2& v) const {
	return Vec2(x*v.x, y*v.y);
}