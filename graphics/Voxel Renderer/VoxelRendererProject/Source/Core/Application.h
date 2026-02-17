#pragma once

#include "Window.h"
#include "../Graphics/Renderer.h"

class Application {
public:
    Application(int width, int height, const wchar_t* title, double m_fixed_timestep = 1.0 / 60);
    ~Application() = default;

    void Run();

private:
    void Update(float deltaTime);
    void GenerateChunk();

    Window m_window;
    Renderer m_renderer;
    bool m_running = true;
    double m_fixed_timestep;

    bool m_terrain_generating = false;
    int m_terrain_radius = 0;
    float lod_dist = 100.0f;
    std::vector<PTP_WORK> m_work_items;
    std::vector<HANDLE> m_work_events;

private:
    std::pair<int, int> m_mousePos;
    Cube* m_cube = nullptr;
};
