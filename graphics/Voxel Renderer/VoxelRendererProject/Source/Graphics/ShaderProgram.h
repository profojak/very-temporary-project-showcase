#pragma once

#include "Device.h"

#include <d3d11.h>
#include <DirectXMath.h>

class ShaderProgram {
private:
    struct constantStruct {
        DirectX::XMMATRIX m_MVP = DirectX::XMMatrixIdentity();
    } m_constantStruct;

public:
    ShaderProgram(Device& device) : m_device(device), m_vertexShader(NULL), m_pixelShader(NULL),
        m_vertexShaderLayout(NULL), m_constantBuffer(NULL) {}
    ~ShaderProgram() {
        m_vertexShader->Release();
        m_pixelShader->Release();
        m_vertexShaderLayout->Release();
        m_constantBuffer->Release();
    }

    const struct constantStruct& GetConstantStruct() const { return m_constantStruct; }
    struct constantStruct& GetConstantStructPtr() { return m_constantStruct; }
    unsigned long long GetConstantStructSize() const { return sizeof(struct constantStruct); }

    ID3D11Buffer* GetConstantBuffer() const { return m_constantBuffer; }
    ID3D11Buffer** GetConstantBufferPtr() { return &m_constantBuffer; }

    void Compile(const wchar_t* vertexPath, const wchar_t* pixelPath);

private:
    Device& m_device;
    ID3D11VertexShader* m_vertexShader;
    ID3D11PixelShader* m_pixelShader;
    ID3D11InputLayout* m_vertexShaderLayout;
    ID3D11Buffer* m_constantBuffer;
};
