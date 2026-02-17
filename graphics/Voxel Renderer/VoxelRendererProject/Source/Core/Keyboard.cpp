#include "Keyboard.h"

bool Keyboard::KeyIsPressed(uchar_t keycode) const noexcept {
    return m_keyStates[keycode];
}

Event Keyboard::ReadKey() noexcept {
    if (!m_keyBuffer.empty()) {
        Event e = m_keyBuffer.front();
        m_keyBuffer.pop();
        return e;
    }

    return Event();
}

bool Keyboard::KeyIsEmpty() const noexcept {
    return m_keyBuffer.empty();
}

char Keyboard::ReadChar() noexcept {
    if (!m_charBuffer.empty()) {
        char character = m_charBuffer.front();
        m_charBuffer.pop();
        return character;
    }

    return 0;
}

bool Keyboard::CharIsEmpty() const noexcept {
    return m_charBuffer.empty();
}

void Keyboard::FlushKey() noexcept {
    m_keyBuffer = std::queue<Event>();
}

void Keyboard::FlushChar() noexcept {
    m_charBuffer = std::queue<char>();
}

void Keyboard::Flush() noexcept {
    FlushKey();
    FlushChar();
}

void Keyboard::SetAutorepeatEnabled(bool enabled) noexcept {
    m_autorepeatEnabled = enabled;
}

bool Keyboard::AutorepeatIsEnabled() const noexcept {
    return m_autorepeatEnabled;
}

void Keyboard::OnKeyEvent(uchar_t keycode, bool pressed) noexcept {
    m_keyStates[keycode] = pressed;
    EventType type = pressed ? EventType::Press : EventType::Release;
    m_keyBuffer.push(Event(type, keycode));
    TrimBuffer(m_keyBuffer);
}

void Keyboard::OnChar(char character) noexcept {
    m_charBuffer.push(character);
    TrimBuffer(m_charBuffer);
}

void Keyboard::ClearState() noexcept {
    m_keyStates.reset();
}

template<typename T>
void Keyboard::TrimBuffer(std::queue<T>& m_buffer) noexcept {
    while (m_buffer.size() > m_bufferSize) {
        m_buffer.pop();
    }
}
