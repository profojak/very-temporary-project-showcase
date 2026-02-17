#pragma once

#include "../Core/WindowsHeaders.h"
#include "Logger.h"

#include <stdexcept>

template<typename... Args>
inline void ThrowIfFailed(HRESULT hr, Args... args) {
    if (FAILED(hr)) {
        LOG_FATAL(args...);
        throw std::runtime_error("Runtime error!");
    }
}

inline void ThrowIfFailed(HRESULT hr) {
    if (FAILED(hr)) {
        throw std::runtime_error("Runtime error!");
    }
}

template<typename... Args>
inline void ThrowIfFailed(BOOL b, Args... args) {
    if (!b) {
        LOG_FATAL(args...);
        throw std::runtime_error("Runtime error!");
    }
}

inline void ThrowIfFailed(BOOL b) {
    if (!b) {
        throw std::runtime_error("Runtime error!");
    }
}
