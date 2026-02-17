/******************************************************************************************
 *	Chili DirectX Framework Version 16.07.20											  *
 *	Mouse.cpp																			  *
 *	Copyright 2016 PlanetChili <http://www.planetchili.net>								  *
 *																						  *
 *	This file is part of The Chili DirectX Framework.									  *
 *																						  *
 *	The Chili DirectX Framework is free software: you can redistribute it and/or modify	  *
 *	it under the terms of the GNU General Public License as published by				  *
 *	the Free Software Foundation, either version 3 of the License, or					  *
 *	(at your option) any later version.													  *
 *																						  *
 *	The Chili DirectX Framework is distributed in the hope that it will be useful,		  *
 *	but WITHOUT ANY WARRANTY; without even the implied warranty of						  *
 *	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the						  *
 *	GNU General Public License for more details.										  *
 *																						  *
 *	You should have received a copy of the GNU General Public License					  *
 *	along with The Chili DirectX Framework.  If not, see <http://www.gnu.org/licenses/>.  *
 ******************************************************************************************/

#include "WindowsHeaders.h"
#include "Mouse.h"

std::pair<int, int> Mouse::GetPos() const noexcept {
    return { m_x, m_y };
}

std::optional<Mouse::RawDelta> Mouse::ReadRawDelta() noexcept {
    if (m_rawDeltaBuffer.empty())
        return std::nullopt;

    const RawDelta d = m_rawDeltaBuffer.front();
    m_rawDeltaBuffer.pop();
    return d;
}

int Mouse::GetPosX() const noexcept {
    return m_x;
}

int Mouse::GetPosY() const noexcept {
    return m_y;
}

bool Mouse::IsInWindow() const noexcept {
    return m_isInWindow;
}

bool Mouse::LeftIsPressed() const noexcept {
    return m_leftIsPressed;
}

bool Mouse::RightIsPressed() const noexcept {
    return m_rightIsPressed;
}

std::optional<Mouse::Event> Mouse::Read() noexcept {
    if (m_buffer.size() > 0u) {
        Mouse::Event e = m_buffer.front();
        m_buffer.pop();
        return e;
    }
    return {};
}

void Mouse::Flush() noexcept {
    m_buffer = std::queue<Event>();
}

void Mouse::EnableRaw() noexcept {
    m_rawEnabled = true;
}

void Mouse::DisableRaw() noexcept {
    m_rawEnabled = false;
}

bool Mouse::RawEnabled() const noexcept {
    return m_rawEnabled;
}

void Mouse::OnMouseMove(int newx, int newy) noexcept {
    m_x = newx;
    m_y = newy;

    m_buffer.push(Mouse::Event(Mouse::Event::Type::Move, *this));
    TrimBuffer();
}

void Mouse::OnMouseLeave() noexcept {
    m_isInWindow = false;
    m_buffer.push(Mouse::Event(Mouse::Event::Type::Leave, *this));
    TrimBuffer();
}

void Mouse::OnMouseEnter() noexcept {
    m_isInWindow = true;
    m_buffer.push(Mouse::Event(Mouse::Event::Type::Enter, *this));
    TrimBuffer();
}

void Mouse::OnRawDelta(int dx, int dy) noexcept {
    m_rawDeltaBuffer.push({ dx,dy });
    TrimBuffer();
}

void Mouse::OnLeftPressed(int x, int y) noexcept {
    m_leftIsPressed = true;

    m_buffer.push(Mouse::Event(Mouse::Event::Type::LPress, *this));
    TrimBuffer();
}

void Mouse::OnLeftReleased(int x, int y) noexcept {
    m_leftIsPressed = false;

    m_buffer.push(Mouse::Event(Mouse::Event::Type::LRelease, *this));
    TrimBuffer();
}

void Mouse::OnRightPressed(int x, int y) noexcept {
    m_rightIsPressed = true;

    m_buffer.push(Mouse::Event(Mouse::Event::Type::RPress, *this));
    TrimBuffer();
}

void Mouse::OnRightReleased(int x, int y) noexcept {
    m_rightIsPressed = false;

    m_buffer.push(Mouse::Event(Mouse::Event::Type::RRelease, *this));
    TrimBuffer();
}

void Mouse::OnWheelUp(int x, int y) noexcept {
    m_buffer.push(Mouse::Event(Mouse::Event::Type::WheelUp, *this));
    TrimBuffer();
}

void Mouse::OnWheelDown(int x, int y) noexcept {
    m_buffer.push(Mouse::Event(Mouse::Event::Type::WheelDown, *this));
    TrimBuffer();
}

void Mouse::TrimBuffer() noexcept {
    while (m_buffer.size() > m_bufferSize) {
        m_buffer.pop();
    }
}

void Mouse::TrimRawInputBuffer() noexcept {
    while (m_rawDeltaBuffer.size() > m_bufferSize) {
        m_rawDeltaBuffer.pop();
    }
}

void Mouse::OnWheelDelta(int x, int y, int delta) noexcept {
    m_wheelDeltaCarry += delta;

    // Generate events for every 120 
    while (m_wheelDeltaCarry >= WHEEL_DELTA) {
        m_wheelDeltaCarry -= WHEEL_DELTA;
        OnWheelUp(x, y);
    }

    while (m_wheelDeltaCarry <= -WHEEL_DELTA) {
        m_wheelDeltaCarry += WHEEL_DELTA;
        OnWheelDown(x, y);
    }
}
