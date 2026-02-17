#pragma once

#include <d3d11.h>

class Device {
public:
    Device() : m_device(NULL), m_deviceContext(NULL) {}
    ~Device() {
        m_device->Release();
        m_deviceContext->Release();
    }

    ID3D11Device* GetDevice() const { return m_device; }
    ID3D11DeviceContext* GetContext() const { return m_deviceContext; }
    ID3D11Device** GetDevicePtr() { return &m_device; }
    ID3D11DeviceContext** GetContextPtr() { return &m_deviceContext; }

private:
    ID3D11Device* m_device;
    ID3D11DeviceContext* m_deviceContext;
};
