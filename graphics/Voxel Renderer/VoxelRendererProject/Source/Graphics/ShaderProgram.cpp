#include "ShaderProgram.h"
#include "../Utilities/Logger.h"
#include "../Utilities/ErrorHandler.h"

#include <d3dcompiler.h>

void ShaderProgram::Compile(const wchar_t* vertexPath, const wchar_t* pixelPath) {
    // Compile vertex shader
    // https://learn.microsoft.com/en-us/windows/win32/direct3d11/how-to--compile-a-shader
    ID3DBlob* shaderBlob = NULL;
    ThrowIfFailed(D3DCompileFromFile(vertexPath, 0, D3D_COMPILE_STANDARD_FILE_INCLUDE,
        "main", "vs_4_0_level_9_1", 0, 0, &shaderBlob, 0),
        "Failed to compile vertex shader!");

    LOG_DEBUG("Compiled vertex shader.");

    ThrowIfFailed(m_device.GetDevice()->CreateVertexShader(
        shaderBlob->GetBufferPointer(), shaderBlob->GetBufferSize(), 0, &m_vertexShader),
        "Failed to create vertex shader!");

    LOG_DEBUG("Created vertex shader.");

    // Create vertex shader input layout
    // https://learn.microsoft.com/en-us/windows/win32/api/d3d11/nf-d3d11-id3d11device-createinputlayout
    D3D11_INPUT_ELEMENT_DESC ied[] = {
        { "POSITION", 0, DXGI_FORMAT_R32_UINT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0 },
        { "COLOR", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 4, D3D11_INPUT_PER_VERTEX_DATA, 0 },
        { "POSITION", 1, DXGI_FORMAT_R32G32B32_FLOAT, 1, 0, D3D11_INPUT_PER_INSTANCE_DATA, 1 },
        { "POSITION", 2, DXGI_FORMAT_R32_UINT, 1, 12, D3D11_INPUT_PER_INSTANCE_DATA, 1 },
        { "SCALE", 0, DXGI_FORMAT_R32G32_UINT, 1, 16, D3D11_INPUT_PER_INSTANCE_DATA, 1 }
    };
    UINT numElements = sizeof(ied) / sizeof(D3D11_INPUT_ELEMENT_DESC);
    ThrowIfFailed(m_device.GetDevice()->CreateInputLayout(
        ied, numElements,
        shaderBlob->GetBufferPointer(),
        shaderBlob->GetBufferSize(),
        &m_vertexShaderLayout),
        "Failed to create vertex input layout!");

    LOG_DEBUG("Created vertex input layout.");

    m_device.GetContext()->IASetInputLayout(m_vertexShaderLayout);
    shaderBlob->Release();

    // Compile pixel shader
    ThrowIfFailed(D3DCompileFromFile(pixelPath, 0, D3D_COMPILE_STANDARD_FILE_INCLUDE,
        "main", "ps_4_0_level_9_1", 0, 0, &shaderBlob, 0),
        "Failed to compile pixel shader!");

    LOG_DEBUG("Compiled pixel shader.");

    ThrowIfFailed(m_device.GetDevice()->CreatePixelShader(
        shaderBlob->GetBufferPointer(), shaderBlob->GetBufferSize(), 0, &m_pixelShader),
        "Failed to create pixel shader!");

    LOG_DEBUG("Created pixel shader.");
    shaderBlob->Release();

    // Set shaders as currently used
    m_device.GetContext()->VSSetShader(m_vertexShader, 0, 0);
    m_device.GetContext()->PSSetShader(m_pixelShader, 0, 0);
}
