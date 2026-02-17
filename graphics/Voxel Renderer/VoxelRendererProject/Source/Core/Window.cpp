#include "Window.h"

#include "../Utilities/Logger.h"
#include "../Utilities/ErrorHandler.h"

/// Static window class instance
Window::WindowClass Window::WindowClass::m_wndClass;

Window::WindowClass::WindowClass() : m_hInst(GetModuleHandle(nullptr)) {
    // Register window class
    // https://learn.microsoft.com/en-us/windows/win32/api/winuser/ns-winuser-wndclassexa
    WNDCLASSEX wc = { 0 };
    wc.cbSize = sizeof(wc);
    wc.style = CS_OWNDC;
    wc.lpfnWndProc = HandleMsgSetup;
    wc.cbClsExtra = 0;
    wc.cbWndExtra = 0;
    wc.hInstance = GetInstance();
    wc.hIcon = nullptr;
    wc.hCursor = nullptr;
    wc.hbrBackground = nullptr;
    wc.lpszMenuName = nullptr;
    wc.lpszClassName = GetName();
    wc.hIconSm = NULL;
    RegisterClassEx(&wc);
}

Window::WindowClass::~WindowClass() {
    UnregisterClass(GetName(), GetInstance());
}

Window::Window(int width, int height, const wchar_t* title):
    m_handle(NULL), m_width(width), m_height(height), m_title(title) {
    // Calculate window size based on desired client region size
    RECT wr = { 0 };
    wr.left = 100;
    wr.right = m_width + wr.left;
    wr.top = 100;
    wr.bottom = m_height + wr.top;
    ThrowIfFailed(AdjustWindowRect(&wr, WS_CAPTION | WS_MINIMIZEBOX | WS_SYSMENU, FALSE),
        "Failed to calculate window size based on input!");

    LOG_DEBUG("Calculated window size based on input.");

    // Create window
    // https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-createwindowexa
    m_handle = CreateWindow(
        WindowClass::GetName(), m_title.data(),
        WS_CAPTION | WS_MINIMIZEBOX | WS_SYSMENU,
        CW_USEDEFAULT, CW_USEDEFAULT, m_width, m_height,
        nullptr, nullptr, WindowClass::GetInstance(), this
    );

    LOG_DEBUG("Created window instance.");
}

Window::~Window() {
    DestroyWindow(m_handle);
}

void Window::Show() {
    // https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-showwindow
    ShowWindow(m_handle, SW_SHOW);

    LOG_DEBUG("Made window visible.");
}

bool Window::ProcessMessages() {
    // https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-peekmessagea
    MSG msg = { 0 };
    BOOL gResult;
    if ((gResult = PeekMessage(&msg, NULL, 0, 0, PM_REMOVE)) > 0) {
        // https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-translatemessage
        TranslateMessage(&msg);
        // https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-dispatchmessage
        DispatchMessage(&msg);

        // see list of available keys in the link below
        // https://learn.microsoft.com/en-us/windows/win32/inputdev/virtual-key-codes
        if (msg.message == WM_QUIT) {
            LOG_DEBUG("Received QUIT message.");
            return false;
        }
    }

    return true;
}

LRESULT CALLBACK Window::HandleMsgSetup(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    if (msg == WM_NCCREATE) {
        const CREATESTRUCTW* const pCreate = reinterpret_cast<CREATESTRUCTW*>(lParam);
        // Retrieve pointer to the window
        Window* const pWnd = static_cast<Window*>(pCreate->lpCreateParams);
        // Set WinAPI-managed user data to store ptr to Window class
        SetWindowLongPtr(hWnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(pWnd));
        // Set message proc to non-setup handler
        SetWindowLongPtr(hWnd, GWLP_WNDPROC, reinterpret_cast<LONG_PTR>(&Window::HandleMsgThunk));
        return pWnd->HandleMsg(hWnd, msg, wParam, lParam);
    }
    return DefWindowProc(hWnd, msg, wParam, lParam);
}

LRESULT CALLBACK Window::HandleMsgThunk(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    Window* const pWnd = reinterpret_cast<Window*>(GetWindowLongPtr(hWnd, GWLP_USERDATA));
    return pWnd->HandleMsg(hWnd, msg, wParam, lParam);
}

LRESULT Window::HandleMsg(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    // https://learn.microsoft.com/en-us/windows/win32/winmsg/about-messages-and-message-queues#system-defined-messages
    // https://www.autohotkey.com/docs/v1/misc/SendMessageList.htm
    switch (msg) {
    case WM_CLOSE: {
        // https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-postquitmessage
        PostQuitMessage(0);
        return 0;
    }

    // This message appears in case focus is lost on the window (like Alt+Tab)
    case WM_KILLFOCUS: {
        // In that case, clear state to not have some phantom pressed keys
        m_keyboard.ClearState();
        break;
    }

    case WM_KEYDOWN: {
        // See lParam desc in the link down below to understand magic constant value
        // https://learn.microsoft.com/en-us/windows/win32/inputdev/wm-keydown
        if (!(lParam & (0x1 << 30)) || m_keyboard.AutorepeatIsEnabled()) {
            m_keyboard.OnKeyEvent(static_cast<uchar_t>(wParam), true);
        }
        break;
    }
    case WM_KEYUP: {
        m_keyboard.OnKeyEvent(static_cast<uchar_t>(wParam), false);
        break;
    }
    case WM_CHAR: {
        m_keyboard.OnChar(static_cast<uchar_t>(wParam));
        break;
    }

    case WM_MOUSEMOVE: {
        const POINTS pt = MAKEPOINTS(lParam);
        if (pt.x >= 0 && pt.x < GetWidth() && pt.y >= 0 && pt.y < GetHeight()) {
            m_mouse.OnMouseMove(pt.x, pt.y);
            if (!m_mouse.IsInWindow()) {
                SetCapture(m_handle);
                m_mouse.OnMouseEnter();
            }
        } else {
            if (wParam & (MK_LBUTTON | MK_RBUTTON)) {
                m_mouse.OnMouseMove(pt.x, pt.y);
            } else {
                ReleaseCapture();
                m_mouse.OnMouseLeave();
            }
        }
        break;
    }

    case WM_LBUTTONDOWN: {
        const POINTS pt = MAKEPOINTS(lParam);
        m_mouse.OnLeftPressed(pt.x, pt.y);
        break;
    }
    case WM_LBUTTONUP: {
        const POINTS pt = MAKEPOINTS(lParam);
        m_mouse.OnLeftReleased(pt.x, pt.y);
        break;
    }
    case WM_RBUTTONDOWN: {
        const POINTS pt = MAKEPOINTS(lParam);
        m_mouse.OnRightPressed(pt.x, pt.y);
        break;
    }
    case WM_RBUTTONUP: {
        const POINTS pt = MAKEPOINTS(lParam);
        m_mouse.OnRightReleased(pt.x, pt.y);
        break;
    }
    case WM_MOUSEWHEEL: {
        const POINTS pt = MAKEPOINTS(lParam);
        const int delta = GET_WHEEL_DELTA_WPARAM(wParam);
        m_mouse.OnWheelDelta(pt.x, pt.y, delta);
        break;
    }
    }

    return DefWindowProc(hWnd, msg, wParam, lParam);
}
