//
// Created by brian on 11/20/18.
//
#pragma once

#include <iostream>
#include <cmath>

const float PI = 3.14159265358979f;

class Complex {
public:
    float real;
    float imag;

    __device__ __host__ Complex() : real(0.0f), imag(0.0f) {}
    __device__ __host__ Complex(float r, float i) : real(r), imag(i) {}
    __device__ __host__ Complex(float r) : real(r), imag(0.0f) {}
    __device__ __host__ Complex operator+(const Complex& b) const { return Complex(b.real+real, b.imag+imag); }
    __device__ __host__ Complex operator-(const Complex& b) const { return Complex(real-b.real, imag-b.imag); }
    __device__ __host__ Complex operator*(const Complex& b) const { return Complex(real*b.real, imag*b.imag); }

    __device__ __host__ Complex mag() const { return sqrt(real*real + imag*imag);}
    __device__ __host__ Complex angle() const { return atan(imag/real); }
    __device__ __host__ Complex conj() const { return Complex(real, (-1)*imag); }

};

std::ostream& operator<< (std::ostream& os, const Complex& rhs) {
    Complex c(rhs);
    if(fabsf(rhs.imag) < 1e-3) c.imag = 0.0f;
    if(fabsf(rhs.real) < 1e-3) c.real = 0.0f;
    if(c.imag==0) os << c.real ;
    else os << "(" << c.real << "," << c.imag << ")";
    return os;
};

