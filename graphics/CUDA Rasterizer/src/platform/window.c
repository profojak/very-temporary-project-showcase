/* CUDA C Software Rendering Pipeline ---------------------------------------
   Bachelor Thesis                                              Jakub Profota
   Summer Semester 2022/3                    OI - Computer Games and Graphics
   Supervisor Jiri Bittner               Czech Technical University in Prague
   window.c ----------------------------------------------------------------- */


#include "window.h"

#include "../core/define.h"
#include "../core/input.h"
#include "print.h"

#include "win32.h"
#include <Windows.h>

#include <stdlib.h> // EXIT_SUCCESS, EXIT_FAILURE


static struct window_t {
    HWND handle;             // Handle to window
    HDC window_dc, frame_dc; // Handle to window and frame device context
    HBITMAP bitmap;          // Handle to bitmap
    BITMAPINFO bi;           // Bitmap info
    PAINTSTRUCT paint;       // Paint
    unsigned int *buffer;    // Framebuffer
    enum status_e status;    // Window status
} window = { 0 };


LRESULT CALLBACK window_procedure(HWND, UINT, WPARAM, LPARAM);

static void window_callback_paint(void);


/* Init and free ------------------------------------------------------------ */


/* Initialize window */
extern int window_init(
    void *handle_instance, // Handle to identify executable
    int show_flag          // Window flag to show or hide window
) {
    // Make window system DPI aware, do not scale when DPI changes
    if (SetProcessDpiAwarenessContext(
        DPI_AWARENESS_CONTEXT_SYSTEM_AWARE
    ) != TRUE) {
        #ifdef _DEBUG
        LOG_ERROR("Failed to configure system DPI awareness!");
        #endif /* _DEBUG */
        return EXIT_FAILURE;
    }

    // Window class
    WNDCLASSEXA wc = { 0 };
    wc.cbSize = sizeof(wc);                    // Size in bytes
    wc.style = 0;                              // Class style
    wc.lpfnWndProc = window_procedure;         // Window procedure
    wc.cbClsExtra = 0;                         // Extra bytes to allocate
    wc.cbWndExtra = 0;                         // Extra bytes to allocate
    wc.hInstance = (HINSTANCE)handle_instance; // Handle to identify executable
    wc.hIcon = 0;                              // Handle to class icon
    wc.hCursor = LoadCursorA(0, IDC_ARROW);    // Handle to class cursor
    wc.hbrBackground = 0;                      // Handle to background brush
    wc.lpszMenuName = 0;                       // No default window menu
    wc.lpszClassName = "cuda_pipeline";        // Window class name
    wc.hIconSm = 0;                            // Handle to small icon

    if (RegisterClassExA(&wc) == 0) {
        #ifdef _DEBUG
        LOG_ERROR("Failed to register window class!");
        #endif /* _DEBUG */
        return EXIT_FAILURE;
    }

    // Window
    DWORD style = WS_CAPTION | WS_SYSMENU | WS_MINIMIZEBOX;
    RECT rect = { 0, 0, WINDOW_WIDTH, WINDOW_HEIGHT };

    if (AdjustWindowRectEx(&rect, style, 0, 0) != TRUE) {
        #ifdef _DEBUG
        LOG_ERROR("Failed to set window size!");
        #endif /* _DEBUG */
        return EXIT_FAILURE;
    }

    int window_width = rect.right - rect.left;
    int window_height = rect.bottom - rect.top;

    window.handle = CreateWindowExA(
        0,                          // Extended window style
        "cuda_pipeline",            // Window class name
        "CUDA",                     // Window name
        style,                      // Window style
        CW_USEDEFAULT,              // Initial horizontal window position
        CW_USEDEFAULT,              // Initial vertical window position
        window_width,               // Window width
        window_height,              // Window height
        0,                          // Handle to parent window
        0,                          // Handle to menu
        (HINSTANCE)handle_instance, // Handle to identify executable
        0                           // Pointer to pass value upon creation
    );

    if (window.handle == 0) {
        #ifdef _DEBUG
        LOG_ERROR("Failed to create window!");
        #endif /* _DEBUG */
        return EXIT_FAILURE;
    }

    // Bitmap info
    window.bi.bmiHeader.biSize = sizeof(
        window.bi.bmiHeader);                     // Number of bytes required
    window.bi.bmiHeader.biWidth = WINDOW_WIDTH;   // Width of bitmap
    window.bi.bmiHeader.biHeight = WINDOW_HEIGHT; // Height of bitmap
    window.bi.bmiHeader.biPlanes = 1;             // Always 1
    window.bi.bmiHeader.biBitCount = 32;          // Bits per pixel
    window.bi.bmiHeader.biCompression = BI_RGB;   // Uncompressed RGB

    // Framebuffer
    window.window_dc = GetDC(window.handle);
    window.bitmap = CreateDIBSection(window.window_dc, &window.bi,
        DIB_RGB_COLORS, (void **)&window.buffer, 0, 0);
    window.frame_dc = CreateCompatibleDC(window.window_dc);
    SelectObject(window.frame_dc, window.bitmap);

    #ifdef _DEBUG
    ASSERT_MSG(((((window.bi.bmiHeader.biWidth * window.bi.bmiHeader.biBitCount)
        + 31) & ~31) >> 3
    ) == 4 * WINDOW_WIDTH, "Stride must be equal to 4x window width!");
    #endif /* _DEBUG */

    // Window and console screen position
    if (global_align) {
        int border_width = GetSystemMetrics(SM_CXBORDER);
        if (SetWindowPos(
            window.handle,            // Handle to window
            0,                        // Handle to window to precede
            2,                        // New X
            (GetSystemMetrics(SM_CYSCREEN) - WINDOW_HEIGHT) / 2, // New Y
            0,                        // New width
            0,                        // New height
            SWP_NOSIZE | SWP_NOZORDER // Retain current size and Z order
        ) != TRUE) {
            #ifdef _DEBUG
            LOG_ERROR("Failed to set window position!");
            #endif /* _DEBUG */
            return EXIT_FAILURE;
        }

        if (SetWindowPos(GetConsoleWindow(), 0,
            WINDOW_WIDTH + 2 * border_width + 2 * 5, 5,
            0, 0, SWP_NOSIZE | SWP_NOZORDER
        ) != TRUE) {
            #ifdef _DEBUG
            LOG_ERROR("Failed to set console window position!");
            #endif /* _DEBUG */
            return EXIT_FAILURE;
        }
    }

    ShowWindow(window.handle, show_flag);
    window_set_status(WINDOW_ACTIVE);

    #ifdef _DEBUG
    LOG_TEXT("Initialized window system");
    LOG_TRACE;

    if (global_verbose) {
        LOG_DEBUG("%u x %u pixels",
            window.bi.bmiHeader.biWidth, window.bi.bmiHeader.biHeight);
    }
    #endif /* _DEBUG */

    return EXIT_SUCCESS;
} /* window_init */


/* Free window */
extern int window_free(
    void
) {
    ReleaseDC(window.handle, window.frame_dc);
    if (DeleteDC(window.frame_dc) != TRUE) {
        #ifdef _DEBUG
        LOG_ERROR("Failed to destroy framebuffer!");
        #endif /* _DEBUG */
        return EXIT_FAILURE;
    }
    window.frame_dc = 0;

    if (DeleteObject(window.bitmap) != TRUE) {
        #ifdef _DEBUG
        LOG_ERROR("Failed to destroy bitmap!");
        #endif /* _DEBUG */
        return EXIT_FAILURE;
    }
    window.bitmap = 0;

    ReleaseDC(window.handle, window.window_dc);
    window.window_dc = 0;

    if (DestroyWindow(window.handle) != TRUE) {
        #ifdef _DEBUG
        LOG_ERROR("Failed to destroy window!");
        #endif /* _DEBUG */
        return EXIT_FAILURE;
    }
    window.handle = 0;
    window_set_status(WINDOW_NOT_INITIALIZED);

    #ifdef _DEBUG
    LOG_TEXT("Freed window system");
    LOG_TRACE;
    #endif /* _DEBUG */

    return EXIT_SUCCESS;
} /* window_free */


/* Procedure ---------------------------------------------------------------- */


/* Window procedure */
LRESULT CALLBACK window_procedure(
    HWND handle,    // Handle to window
    UINT message,   // Message
    WPARAM w_param, // Additional message information
    LPARAM l_param  // Additional message information
) {
    switch (message) {

    // Close button
    case WM_CLOSE: {
        window_set_status(WINDOW_SHUTTING_DOWN);
        return 0;
    } break;

    // Post WM_QUIT message to thread message queue when destroyed
    case WM_DESTROY: {
        PostQuitMessage(0);
    } break;

    // Key press
    case WM_KEYDOWN: {
        input_down(w_param, l_param & (1 << 30));
    } break;

    // Key release
    case WM_KEYUP: {
        input_up(w_param);
    } break;

    // Draw window contents
    case WM_PAINT: {
        window_callback_paint();
    } break;

    }
    return DefWindowProcA(handle, message, w_param, l_param);
} /* window_procedure */


/* Callback ----------------------------------------------------------------- */


/* Process incoming message */
extern inline void window_callback_message(
    void
) {
    MSG message;
    while (PeekMessageA(
        &message,      // Message structure receiving message information
        window.handle, // Receive from any window belonging to current thread
        0,             // Return all available messages
        0,             // Return all available messages
        PM_REMOVE      // Remove from queue after processing
    )) {
        // Translate virtual keys to characters
        TranslateMessage(&message);

        // Dispatch message to window procedure
        DispatchMessage(&message);
    }
} /* window_callback_message */


/* Invalidate window screen and raise WM_PAINT message */
extern inline void window_callback_invalidate(
    void
) {
    InvalidateRect(
        window.handle, // Handle to window to update
        0,             // Update entire client area
        FALSE          // Background not erased
    );
} /* window_callback_update */


/* Draw window contents */
static inline void window_callback_paint(
    void
) {
    HDC device_context = BeginPaint(window.handle, &window.paint);
    BitBlt(
        device_context,  // Handle to destination device context
        0,               // X coordinate of upper left corner of destination
        0,               // Y coordinate of upper left corner of destination
        WINDOW_WIDTH,    // Width of source and destination
        WINDOW_HEIGHT,   // Height of source and destination
        window.frame_dc, // Handle to source device context
        0,               // X coordinate of upper left corner of source
        0,               // Y coordinate of upper left corner of source
        SRCCOPY          // Copy source directly to destination
    );
    EndPaint(window.handle, &window.paint);
} /* window_callback_paint */


/* Set ---------------------------------------------------------------------- */


/* Set window status */
extern inline void window_set_status(
    enum status_e status // Status
) {
    window.status = status;
} /* window_set_status */


/* Get ---------------------------------------------------------------------- */


/* Get window status */
extern inline enum status_e window_get_status(
    void
) {
    return window.status;
} /* window_get_status */


/* Get window bitmap buffer */
extern inline unsigned int **window_get_buffer(
    void
) {
    return &window.buffer;
} /* window_get_buffer */


/* -------------------------------------------------------------------------- */


/* window.c */