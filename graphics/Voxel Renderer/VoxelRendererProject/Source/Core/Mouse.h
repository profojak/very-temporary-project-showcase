/******************************************************************************************
 *	Chili DirectX Framework Version 16.07.20											  *
 *	Mouse.h																				  *
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

#pragma once

#include <queue>
#include <optional>

class Mouse {
    friend class Window;
public:
    struct RawDelta {
        int x, y;
    };

    class Event {
    public:
        enum class Type {
            LPress,
            LRelease,
            RPress,
            RRelease,
            WheelUp,
            WheelDown,
            Move,
            Enter,
            Leave,
        };

    private:
        Type m_type;
        bool m_leftIsPressed;
        bool m_rightIsPressed;
        int m_x;
        int m_y;

    public:
        Event(Type type, const Mouse& parent) noexcept :
            m_type(type), m_leftIsPressed(parent.m_leftIsPressed),
            m_rightIsPressed(parent.m_rightIsPressed), m_x(parent.m_x), m_y(parent.m_y) {}

        Type GetType() const noexcept { return m_type; }
        std::pair<int, int> GetPos() const noexcept { return{ m_x, m_y }; }
        int GetPosX() const noexcept { return m_x; }
        int GetPosY() const noexcept { return m_y; }

        bool LeftIsPressed() const noexcept { return m_leftIsPressed; }
        bool RightIsPressed() const noexcept { return m_rightIsPressed; }
    };

public:
    Mouse() = default;
    Mouse(const Mouse&) = delete;

    Mouse& operator=(const Mouse&) = delete;

    std::pair<int, int> GetPos() const noexcept;
    std::optional<RawDelta> ReadRawDelta() noexcept;
    int GetPosX() const noexcept;
    int GetPosY() const noexcept;

    bool IsInWindow() const noexcept;
    bool LeftIsPressed() const noexcept;
    bool RightIsPressed() const noexcept;
    std::optional<Mouse::Event> Read() noexcept;

    bool IsEmpty() const noexcept { return m_buffer.empty(); }
    void Flush() noexcept;
    void EnableRaw() noexcept;
    void DisableRaw() noexcept;
    bool RawEnabled() const noexcept;

private:
    void OnMouseMove(int x, int y) noexcept;
    void OnMouseLeave() noexcept;
    void OnMouseEnter() noexcept;
    void OnRawDelta(int dx, int dy) noexcept;
    void OnLeftPressed(int x, int y) noexcept;
    void OnLeftReleased(int x, int y) noexcept;
    void OnRightPressed(int x, int y) noexcept;
    void OnRightReleased(int x, int y) noexcept;
    void OnWheelUp(int x, int y) noexcept;
    void OnWheelDown(int x, int y) noexcept;
    void TrimBuffer() noexcept;
    void TrimRawInputBuffer() noexcept;
    void OnWheelDelta(int x, int y, int delta) noexcept;

private:
    static constexpr unsigned int m_bufferSize = 16u;
    int m_x;
    int m_y;
    bool m_leftIsPressed = false;
    bool m_rightIsPressed = false;
    bool m_isInWindow = false;
    int m_wheelDeltaCarry = 0;
    bool m_rawEnabled = false;
    std::queue<Event> m_buffer;
    std::queue<RawDelta> m_rawDeltaBuffer;
};
