#pragma once

#include <DirectXMath.h>

class Camera {
public:
    Camera(int width, int height);
    ~Camera() = default;

    const DirectX::XMMATRIX& GetProjection() const { return m_projection; }
    DirectX::XMMATRIX GetView() const;

    DirectX::XMVECTOR m_position;

    float GetPitchAngle() const { return m_pitch; }
    float GetYawAngle() const { return m_yaw; }
    void SetPitchAngle(float angle);
    void SetYawAngle(float angle);

    DirectX::XMVECTOR GetForwardVector();
    DirectX::XMVECTOR GetRightVector();
    DirectX::XMVECTOR GetUpVector();

private:
    DirectX::XMMATRIX m_projection;

    float m_pitch = 0;
    float m_yaw = 0;

    DirectX::XMVECTOR m_forward;
    DirectX::XMVECTOR m_up;
};
