#pragma once

#include <queue>
#include <bitset>

enum class EventType {
    Press,
    Release,
    Invalid,
};

class Event {
public:
    Event() : m_type(EventType::Invalid), m_code(0u) {}
    Event(EventType type, unsigned char code) noexcept : m_type(type), m_code(code) {}

    bool IsPressed() const noexcept { return m_type == EventType::Press; }
    bool IsReleased() const noexcept { return m_type == EventType::Release; }
    bool IsValid() const noexcept { return m_type == EventType::Invalid; }

    unsigned char GetCode() const noexcept { return m_code; }

private:
    EventType m_type;
    unsigned char m_code;
};

using uchar_t = unsigned char;

class Keyboard {
    friend class Window;

public:
    Keyboard() = default;
    Keyboard(const Keyboard&) = delete;
    Keyboard& operator=(const Keyboard&) = delete;

    bool KeyIsPressed(uchar_t keycode) const noexcept;
    Event ReadKey() noexcept;
    bool KeyIsEmpty() const noexcept;
    void FlushKey() noexcept;

    char ReadChar() noexcept;
    bool CharIsEmpty() const noexcept;
    void FlushChar() noexcept;

    void Flush() noexcept;
    void ClearState() noexcept;

    void SetAutorepeatEnabled(bool enabled) noexcept;
    bool AutorepeatIsEnabled() const noexcept;

private:
    void OnKeyEvent(uchar_t keycode, bool pressed) noexcept;
    void OnChar(char character) noexcept;

    template<typename T>
    static void TrimBuffer(std::queue<T>& m_buffer) noexcept;

private:
    static constexpr unsigned int m_nKeys = 256u;
    static constexpr unsigned int m_bufferSize = 16u;

    bool m_autorepeatEnabled = false;

    std::bitset<m_nKeys> m_keyStates;
    std::queue<Event> m_keyBuffer;
    std::queue<char> m_charBuffer;
};
