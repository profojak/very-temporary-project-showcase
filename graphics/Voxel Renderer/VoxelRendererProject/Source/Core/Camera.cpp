#include "Camera.h"

#include <algorithm>

using namespace DirectX;

Camera::Camera(int width, int height) {
    m_projection = XMMatrixPerspectiveFovLH(0.4f * XM_PI, (float)width / height, 1.0f, 1000.0f);

    m_position = XMVectorSet(1.0f, 2.5f, -8.0f, 0.0f);
    m_up = XMVectorSet(0.0f, 1.0f, 0.0f, 0.0f);
    m_forward = XMVectorSet(0.0f, 0.0f, 1.0f, 0.0f);
}

XMMATRIX Camera::GetView() const {
    auto pitchRad = m_pitch * DirectX::XM_PI / 180.0f;
    auto yawRad = m_yaw * DirectX::XM_PI / 180.0f;
    XMMATRIX camRotationMat = XMMatrixRotationRollPitchYaw(pitchRad, yawRad, 0.0f);
    XMVECTOR camDirection = XMVector3TransformCoord(m_forward, camRotationMat);
    XMVECTOR target = XMVectorAdd(m_position, camDirection);

    return XMMatrixLookAtLH(m_position, target, m_up);
}

void Camera::SetPitchAngle(float angle) {
    m_pitch = std::clamp(angle, -89.0f, 89.0f);
}

void Camera::SetYawAngle(float angle) {
    m_yaw = angle;
    if (m_yaw > 360.0f) { m_yaw -= 360.0f; }
    else if (m_yaw < 0.0f) { m_yaw += 360.0f; }
}

XMVECTOR Camera::GetForwardVector() {
    auto pitchRad = m_pitch * DirectX::XM_PI / 180.0f;
    auto yawRad = m_yaw * DirectX::XM_PI / 180.0f;
    XMMATRIX camRotationMat = XMMatrixRotationRollPitchYaw(pitchRad, yawRad, 0.0f);
    return XMVector3TransformCoord(m_forward, camRotationMat);
}

XMVECTOR Camera::GetRightVector() {
    auto yawRad = m_yaw * DirectX::XM_PI / 180.0f;
    const auto rotMat = XMMatrixRotationRollPitchYaw(0.0f, yawRad, 0.0f);
    return XMVector3Transform({1.0f, 0.0f, 0.0f}, rotMat);
}

XMVECTOR Camera::GetUpVector() {
    return XMVector3Cross(GetForwardVector(), GetRightVector());
}
