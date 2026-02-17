#pragma once

#include <d3d11.h>

class RenderTarget {
public:
    RenderTarget() : m_frameBuffer(NULL), m_targetView(NULL), m_depthStencilBuffer(NULL),
        m_depthStencilView(NULL) {}
    ~RenderTarget() {
        m_frameBuffer->Release();
        m_targetView->Release();
        m_depthStencilBuffer->Release();
        m_depthStencilView->Release();
    }

    ID3D11Texture2D* GetFrameBuffer() const { return m_frameBuffer; }
    ID3D11RenderTargetView* GetTargetView() const { return m_targetView; }
    ID3D11Texture2D* GetDepthStencilBuffer() const { return m_depthStencilBuffer; }
    ID3D11DepthStencilView* GetDepthStencilView() const { return m_depthStencilView; }
    ID3D11Texture2D** GetFrameBufferPtr() { return &m_frameBuffer; }
    ID3D11RenderTargetView** GetTargetViewPtr() { return &m_targetView; }
    ID3D11Texture2D** GetDepthStencilBufferPtr() { return &m_depthStencilBuffer; }
    ID3D11DepthStencilView** GetDepthStencilViewPtr() { return &m_depthStencilView; }

private:
    ID3D11Texture2D* m_frameBuffer;
    ID3D11RenderTargetView* m_targetView;
    ID3D11Texture2D* m_depthStencilBuffer;
    ID3D11DepthStencilView* m_depthStencilView;
};
