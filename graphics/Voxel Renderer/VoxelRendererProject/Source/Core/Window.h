#pragma once

#include "WindowsHeaders.h"
#include "Keyboard.h"
#include "Mouse.h"

#include <string>

class Window {
public:
    Window(int width, int height, const wchar_t* title);
    ~Window();

    HWND GetHandle() const { return m_handle; }
    int GetWidth() const { return m_width; }
    int GetHeight() const { return m_height; }
    const Keyboard& GetKeyboard() const { return m_keyboard; }
    Keyboard& GetKeyboard() { return m_keyboard; }
    const Mouse& GetMouse() const { return m_mouse; }

    void Show();
    /// Returns false when window is closed
    bool ProcessMessages();

private:
    class WindowClass {
    public:
        WindowClass();
        ~WindowClass();

        /// Returns the name of the registered window class
        static const wchar_t* GetName() noexcept { return m_wndClassName; }
        /// Returns the instance handle of the application
        static HINSTANCE GetInstance() noexcept { return m_wndClass.m_hInst; }

    private:
        static constexpr const wchar_t* m_wndClassName = L"WindowClass";
        static WindowClass m_wndClass;
        HINSTANCE m_hInst;
    };

    /// Message handler, sets up the window instance
    static LRESULT CALLBACK HandleMsgSetup(
        HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);
    /// Message handler, redirects messages to the instance-specific message handler
    static LRESULT CALLBACK HandleMsgThunk(
        HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);
    /// Message handler, processes window messages
    LRESULT HandleMsg(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

    HWND m_handle;
    int m_width;
    int m_height;
    std::wstring m_title;
    Keyboard m_keyboard;
    Mouse m_mouse;
};
