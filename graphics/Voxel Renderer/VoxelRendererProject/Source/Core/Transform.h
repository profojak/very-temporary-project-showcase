#pragma once

template <typename T>
struct Vector2 {
    T x = 0.f;
    T y = 0.f;
};

template <typename T>
struct Vector3 {
    T x = 0.f;
    T y = 0.f;
    T z = 0.f;
};

struct Transform {
    Vector3<float> Position;
    Vector3<float> Rotation;
    Vector3<float> Scale = { 1.f, 1.f, 1.f };
};
