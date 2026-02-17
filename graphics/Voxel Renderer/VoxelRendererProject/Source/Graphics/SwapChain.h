#pragma once

#include <d3d11.h>

class SwapChain {
public:
    SwapChain() : m_swapChain(NULL) {}
    ~SwapChain() {
        m_swapChain->Release();
    }

    IDXGISwapChain* Get() const { return m_swapChain; }
    IDXGISwapChain** GetPtr() { return &m_swapChain; }

private:
    IDXGISwapChain* m_swapChain;
};
