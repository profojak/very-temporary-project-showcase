

#include "Renderer.h"
#include "../Utilities/Logger.h"
#include "../Utilities/ErrorHandler.h"

#include <d3d11.h>

Renderer::Renderer(Window& window): m_window(window), m_shaderProgram(m_device),
    m_camera(window.GetWidth(), window.GetHeight()) {
    // Initiliaze Direct3D
    // https://learn.microsoft.com/en-us/windows/win32/api/d3d11/nf-d3d11-d3d11createdeviceandswapchain
    DXGI_SWAP_CHAIN_DESC scd;
    ZeroMemory(&scd, sizeof(DXGI_SWAP_CHAIN_DESC));
    scd.BufferCount = 1;
    scd.BufferDesc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    scd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    scd.SampleDesc.Count = 1;
    scd.OutputWindow = window.GetHandle();
    scd.Windowed = TRUE;
    ThrowIfFailed(D3D11CreateDeviceAndSwapChain(
        0, D3D_DRIVER_TYPE_HARDWARE, 0, D3D11_CREATE_DEVICE_BGRA_SUPPORT, 0, 0, D3D11_SDK_VERSION,
        &scd, m_swapChain.GetPtr(), m_device.GetDevicePtr(), 0, m_device.GetContextPtr()),
        "Failed to initialize Direct3D!");

    LOG_DEBUG("Initialized Direct3D.");

    // Initialize render target and frame buffer
    // https://learn.microsoft.com/en-us/windows/win32/api/d3d11/nf-d3d11-id3d11device-createrendertargetview
    // https://learn.microsoft.com/en-us/windows/win32/api/d3d11/nf-d3d11-id3d11devicecontext-omsetrendertargets
    m_swapChain.Get()->GetBuffer(0, __uuidof(ID3D11Texture2D), (LPVOID*)m_renderTarget.GetFrameBufferPtr());
    if (m_renderTarget.GetFrameBuffer()) {
        m_device.GetDevice()->CreateRenderTargetView(m_renderTarget.GetFrameBuffer(),
            NULL, m_renderTarget.GetTargetViewPtr());
    } else {
        ThrowIfFailed(false, "Failed to initialize render target and frame buffer!");
    }

    LOG_DEBUG("Initialized render target and frame buffer.");

    // Initialize fill mode raster states
    solid_rd.FillMode = D3D11_FILL_SOLID;
    solid_rd.CullMode = D3D11_CULL_NONE;
    solid_rd.FrontCounterClockwise = false;
    solid_rd.DepthClipEnable = true;
    solid_rd.ScissorEnable = false;
    solid_rd.MultisampleEnable = false;
    solid_rd.AntialiasedLineEnable = false;
    HRESULT hr = m_device.GetDevice()->CreateRasterizerState(&solid_rd, &solid_rs);
    ThrowIfFailed(hr, "Failed to create solid fill mode raster state!");

    wireframe_rd.FillMode = D3D11_FILL_WIREFRAME;
    wireframe_rd.CullMode = D3D11_CULL_NONE;
    wireframe_rd.FrontCounterClockwise = false;
    wireframe_rd.DepthClipEnable = true;
    wireframe_rd.ScissorEnable = false;
    wireframe_rd.MultisampleEnable = false;
    wireframe_rd.AntialiasedLineEnable = false;
    hr = m_device.GetDevice()->CreateRasterizerState(&wireframe_rd, &wireframe_rs);
    ThrowIfFailed(hr, "Failed to create wireframe fill mode raster state!");

    m_device.GetContext()->RSSetState(solid_rs.Get());

    LOG_DEBUG("Initialized fill mode raster states.");

    // Initialize depth stencil buffer
    D3D11_TEXTURE2D_DESC td;
    ZeroMemory(&td, sizeof(D3D11_TEXTURE2D_DESC));
    td.Width = m_window.GetWidth();
    td.Height = m_window.GetHeight();
    td.MipLevels = 1;
    td.ArraySize = 1;
    td.Format = DXGI_FORMAT_D24_UNORM_S8_UINT;
    td.SampleDesc.Count = 1;
    td.SampleDesc.Quality = 0;
    td.Usage = D3D11_USAGE_DEFAULT;
    td.BindFlags = D3D11_BIND_DEPTH_STENCIL;

    ThrowIfFailed(m_device.GetDevice()->CreateTexture2D(&td,
        NULL, m_renderTarget.GetDepthStencilBufferPtr()),
        "Failed to create depth stencil buffer!");
    ThrowIfFailed(m_device.GetDevice()->CreateDepthStencilView(m_renderTarget.GetDepthStencilBuffer(),
        NULL, m_renderTarget.GetDepthStencilViewPtr()),
        "Failed to create depth stencil view!");

    m_device.GetContext()->OMSetRenderTargets(1, m_renderTarget.GetTargetViewPtr(),
        m_renderTarget.GetDepthStencilView());

    LOG_DEBUG("Initialized depth stencil buffer.");

    // Initialize shader constant buffer
    D3D11_BUFFER_DESC bd;
    ZeroMemory(&bd, sizeof(D3D11_BUFFER_DESC));
    bd.Usage = D3D11_USAGE_DEFAULT;
    bd.ByteWidth = (UINT)m_shaderProgram.GetConstantStructSize();
    bd.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    ThrowIfFailed(m_device.GetDevice()->CreateBuffer(&bd, NULL, m_shaderProgram.GetConstantBufferPtr()),
        "Failed to create shader constant buffer!");

    LOG_DEBUG("Initialized shader constant buffer.");

    // Set viewport
    // https://learn.microsoft.com/en-us/windows/win32/api/d3d11/nf-d3d11-id3d11devicecontext-rssetviewports
    D3D11_VIEWPORT viewport;
    ZeroMemory(&viewport, sizeof(D3D11_VIEWPORT));
    viewport.TopLeftX = 0;
    viewport.TopLeftY = 0;
    viewport.Width = (FLOAT)window.GetWidth();
    viewport.Height = (FLOAT)window.GetHeight();
    viewport.MinDepth = 0.0f;
    viewport.MaxDepth = 1.0f;
    m_device.GetContext()->RSSetViewports(1, &viewport);

    LOG_DEBUG("Set viewport.");

    InitializeSRWLock(&m_mesh_lock);

    // Initialize text rendering
    D2D1_FACTORY_OPTIONS d2d_options = {};
    hr = D2D1CreateFactory(D2D1_FACTORY_TYPE_SINGLE_THREADED, __uuidof(m_d2d_factory), &d2d_options,
        reinterpret_cast<void**>(m_d2d_factory.GetAddressOf()));
    ThrowIfFailed(hr, "Failed to create D2D factory!");

    Microsoft::WRL::ComPtr<IDXGISurface> dxgi_back_buffer;
    hr = m_swapChain.Get()->GetBuffer(0, __uuidof(IDXGISurface), (void**)dxgi_back_buffer.GetAddressOf()),
    ThrowIfFailed(hr, "Failed to get DXGI back buffer!");

    D2D1_RENDER_TARGET_PROPERTIES props = D2D1::RenderTargetProperties(
        D2D1_RENDER_TARGET_TYPE_HARDWARE, D2D1::PixelFormat(DXGI_FORMAT_UNKNOWN, D2D1_ALPHA_MODE_PREMULTIPLIED));
    hr = m_d2d_factory->CreateDxgiSurfaceRenderTarget(dxgi_back_buffer.Get(), &props, m_d2d_render_target.GetAddressOf());
    ThrowIfFailed(hr, "Failed to create D2D render target!");

    IDWriteFactory* raw_factory = nullptr;
    hr = DWriteCreateFactory(DWRITE_FACTORY_TYPE_SHARED, __uuidof(IDWriteFactory),
        reinterpret_cast<IUnknown**>(&raw_factory));
    ThrowIfFailed(hr, "Failed to create DWrite factory!");
    m_dwrite_factory.Attach(raw_factory);

    hr = m_dwrite_factory->CreateTextFormat(L"Segoe UI", nullptr,
        DWRITE_FONT_WEIGHT_NORMAL, DWRITE_FONT_STYLE_NORMAL, DWRITE_FONT_STRETCH_NORMAL, 24.0f, L"en-us",
        m_text_format.GetAddressOf());
    ThrowIfFailed(hr, "Failed to create text format!");

    hr = m_d2d_render_target->CreateSolidColorBrush(D2D1::ColorF(D2D1::ColorF::White),
        m_text_brush.GetAddressOf());
    ThrowIfFailed(hr, "Failed to create text brush!");
}

void Renderer::Initialize() {
    m_shaderProgram.Compile(L"Source/Shaders/Vertex.hlsl", L"Source/Shaders/Pixel.hlsl");
}

void Renderer::Render(float fps, int chunks, float lod_dist) {
    // Clear frame buffer
    // https://learn.microsoft.com/en-us/windows/win32/api/d3d11/nf-d3d11-id3d11devicecontext-clearrendertargetview
    float clearColor[] = { 0.0f, 0.2f, 0.2f, 1.0f };
    m_device.GetContext()->ClearRenderTargetView(m_renderTarget.GetTargetView(), clearColor);
    m_device.GetContext()->ClearDepthStencilView(m_renderTarget.GetDepthStencilView(),
        D3D11_CLEAR_DEPTH | D3D11_CLEAR_STENCIL, 1.0f, 0);

    AcquireSRWLockExclusive(&m_mesh_lock);
    for (auto& mesh : meshes) {
        const Vector3<float>& position = mesh->transform.Position;
        const Vector3<float>& rotation = mesh->transform.Rotation;
        const Vector3<float>& scale = mesh->transform.Scale;

        DirectX::XMMATRIX model = DirectX::XMMatrixScaling(scale.x, scale.y, scale.z)
            * DirectX::XMMatrixRotationY(rotation.y)
            * DirectX::XMMatrixTranslation(position.x, position.y, position.z);

        auto& constantStruct = m_shaderProgram.GetConstantStructPtr();
        constantStruct.m_MVP = DirectX::XMMatrixTranspose(model * m_camera.GetView() * m_camera.GetProjection() );

        m_device.GetContext()->UpdateSubresource(m_shaderProgram.GetConstantBuffer(), 0, NULL,
            &m_shaderProgram.GetConstantStruct(), 0, 0);
        m_device.GetContext()->VSSetConstantBuffers(0, 1, m_shaderProgram.GetConstantBufferPtr());

        mesh->Draw(m_device, GetCamera(), lod_dist);
    }
    ReleaseSRWLockExclusive(&m_mesh_lock);

    // Render text
    m_d2d_render_target->BeginDraw();
    std::wostringstream wss;
    wss << L"FPS: " << static_cast<int>(fps) << std::endl << L"Chunks: " << chunks
        << std::endl << std::endl
        << L"W,A,S,D,E,Q - camera" << std::endl
        << L"G - generate terrain" << std::endl
        << L"1 - LOD dist. 15" << std::endl << L"2 - LOD dist. 100" << std::endl << L"3 - LOD dist. 1500" << std::endl
        << L"4,5 - solid, wireframe" << std::endl;
    D2D1_RECT_F layout = D2D1::RectF(10, 10, 400, 50);
    m_d2d_render_target->DrawText(wss.str().c_str(), (UINT)wss.str().length(),
        m_text_format.Get(), layout, m_text_brush.Get());
    HRESULT hr = m_d2d_render_target->EndDraw();
    ThrowIfFailed(hr, "Failed to render text!");

    // Switch back and front buffers
    m_swapChain.Get()->Present(0, 0);
}

size_t Renderer::AddMesh(Mesh* mesh) {
    AcquireSRWLockExclusive(&m_mesh_lock);
    meshes.push_back(std::shared_ptr<Mesh>(mesh));
    size_t size = meshes.size() - 1;
    ReleaseSRWLockExclusive(&m_mesh_lock);
    return size;
}

Mesh* Renderer::GetMesh(size_t index) {
    return (index < meshes.size()) ? meshes[index].get() : nullptr;
}

void Renderer::RemoveMesh(size_t id) {
    meshes.erase(meshes.begin() + id);
}
