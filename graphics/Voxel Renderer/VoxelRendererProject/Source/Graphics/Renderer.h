#pragma once

#include "../Core/Window.h"
#include "Device.h"
#include "SwapChain.h"
#include "RenderTarget.h"
#include "ShaderProgram.h"
#include "Primitives.h"
#include "../Core/Camera.h"

#include <memory>
#include <d2d1_1.h>
#include <dwrite.h>
#include <wrl/client.h>

class Renderer {

public:
    Renderer(Window& window);
    ~Renderer() = default;

    void Initialize();
    void Render(float fps, int chunks, float lod_dist);

    size_t AddMesh(Mesh* mesh);
    Mesh* GetMesh(size_t index);
    void RemoveMesh(size_t id);

    Camera& GetCamera() { return m_camera; }
    Device& GetDevice() { return m_device; }

    Microsoft::WRL::ComPtr<ID3D11RasterizerState> solid_rs, wireframe_rs;

private:
    Window&       m_window;
    Device        m_device;
    SwapChain     m_swapChain;
    RenderTarget  m_renderTarget;
    ShaderProgram m_shaderProgram;
    Camera        m_camera;

    std::vector<std::shared_ptr<Mesh>> meshes;

    SRWLOCK m_mesh_lock;

    Microsoft::WRL::ComPtr<ID2D1Factory1> m_d2d_factory;
    Microsoft::WRL::ComPtr<ID2D1RenderTarget> m_d2d_render_target;
    Microsoft::WRL::ComPtr<IDWriteFactory> m_dwrite_factory;
    Microsoft::WRL::ComPtr<IDWriteTextFormat> m_text_format;
    Microsoft::WRL::ComPtr<ID2D1SolidColorBrush> m_text_brush;

    D3D11_RASTERIZER_DESC solid_rd, wireframe_rd;
};
