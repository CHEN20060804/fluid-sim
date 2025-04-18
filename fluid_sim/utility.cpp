#include "utility.h"
#include <cmath>  

//���캯��
Vec2::Vec2(float x, float y) : x(x), y(y) {}
Vec2::Vec2() : x(0.0f), y(0.0f) {} // Ĭ�Ϲ��캯������ʼ��Ϊ(0, 0)


float& Vec2::getX()  {
    return x;
}
float& Vec2::getY()  {
    return y;
}

// �ӷ����������
Vec2 Vec2::operator+(const Vec2& b) const {
    return Vec2(x + b.x, y + b.y);
}

// �������������
Vec2 Vec2::operator-(const Vec2& b) const {
    return Vec2(x - b.x, y - b.y);
}
// �������������
Vec2 Vec2::operator-(){
	return Vec2(-x, -y);
}


// �������������
Vec2 Vec2::operator/(float s) const {
    if (s != 0) {
        return Vec2(x / s, y / s);
    }
    // �������0������������׳��쳣�򷵻�һ��Ĭ�ϵ�Vec2
    return Vec2(0.0f, 0.0f);
}

// �˷����������
Vec2 Vec2::operator*(float s) const {
    return Vec2(x * s, y * s);
}

// �ӷ���ֵ���������
Vec2& Vec2::operator+=(const Vec2& b) {
    x += b.x;
    y += b.y;
    return *this;
}

// ������ֵ���������
Vec2& Vec2::operator-=(const Vec2& b) {
    x -= b.x;
    y -= b.y;
    return *this;
}

// �˷���ֵ���������
Vec2& Vec2::operator*=(float s) {
    x *= s;
    y *= s;
    return *this;
}

// ������ֵ���������
Vec2& Vec2::operator/=(float s) {
    if (s != 0) {
        x /= s;
        y /= s;
    }
    else {
        // ���Կ����׳��쳣������������
        x = y = 0.0f;
    }
    return *this;
}

// ��ȡ�����ĳ���
float Vec2::length() const {
    return std::sqrt(x * x + y * y);
}

// ��һ����������
Vec2 Vec2::normalize() const {
   float len = length();
   if (len > 0) {
       return Vec2(x / len, y / len); // ���ع�һ�����������
   }
   // �������Ϊ0�����������󣬿��Է���һ��Ĭ�ϵ�Vec2
   return Vec2(0.0f, 0.0f);
}

//����<<�����
std::ostream& operator<<(std::ostream& os, const Vec2& vec) {
	os << "(" << vec.x << ", " << vec.y << ")";
	return os;
}

//�������������
Vec2 operator*(float s, Vec2 v) {
    return v.operator*(s);
}

}

Vec2 Vec2::cwiseProduct(const Vec2& v) const {
	return Vec2(x*v.x, y*v.y);
}