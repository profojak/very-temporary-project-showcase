#include "Application.h"
#include "../Utilities/Logger.h"
#include "../Graphics/VoxelChunk.h"

Application::Application(int width, int height, const wchar_t* title, double m_fixed_timestep):
    m_window(width, height, title), m_renderer(m_window), m_fixed_timestep(m_fixed_timestep) {
    m_renderer.Initialize();
    m_window.Show();
}

void Application::Run() {
    std::chrono::time_point<std::chrono::system_clock> first = std::chrono::system_clock::now();
    double accumulator = 0.0;

    while (1) {
        if (!m_window.ProcessMessages())
            break;
        std::chrono::time_point<std::chrono::system_clock> second = std::chrono::system_clock::now();
        std::chrono::duration<double> frame_time = second - first;
        first = second;
        auto delta = frame_time.count();

        // Guard against very large frame times
        if (delta > 0.25)
            delta = 0.25;

        accumulator += delta;
        while (accumulator >= m_fixed_timestep) {
            Update(static_cast<float>(m_fixed_timestep));
            accumulator -= m_fixed_timestep;
        }

        m_renderer.Render(1.0f / static_cast<float>(delta), 4 * m_terrain_radius * m_terrain_radius, lod_dist);
    }
}

void Application::Update(float deltaTime) {
#pragma region Camera_Rotation
    static float camRotationSpeed = 10.0f;
    auto newMousePos = m_window.GetMouse().GetPos();
    if (m_window.GetMouse().LeftIsPressed()) {
        int X = m_mousePos.first - newMousePos.first;
        int Y = m_mousePos.second - newMousePos.second;
        float yaw = -deltaTime * (X * camRotationSpeed) + m_renderer.GetCamera().GetYawAngle();
        float pitch = -deltaTime * (Y * camRotationSpeed) + m_renderer.GetCamera().GetPitchAngle();
        m_renderer.GetCamera().SetYawAngle(yaw);
        m_renderer.GetCamera().SetPitchAngle(pitch);
    }
    m_mousePos = newMousePos;
#pragma endregion Camera_Rotation

#pragma region Camera_Movement
    static float cameraSpeed = 25.0f;
    float forwardMult = 0.0f;
    if (m_window.GetKeyboard().KeyIsPressed('W')) { forwardMult += cameraSpeed; }
    if (m_window.GetKeyboard().KeyIsPressed('S')) { forwardMult -= cameraSpeed; }

    float rightMult = 0.0f;
    if (m_window.GetKeyboard().KeyIsPressed('D')) { rightMult += cameraSpeed; }
    if (m_window.GetKeyboard().KeyIsPressed('A')) { rightMult -= cameraSpeed; }

    float upMult = 0.0f;
    if (m_window.GetKeyboard().KeyIsPressed('E')) { upMult += cameraSpeed; }
    if (m_window.GetKeyboard().KeyIsPressed('Q')) { upMult -= cameraSpeed; }

    auto& cameraPos = m_renderer.GetCamera().m_position;

    DirectX::XMVECTOR cameraForwardDir = DirectX::XMVectorMultiply(
        m_renderer.GetCamera().GetForwardVector(), {
            deltaTime * forwardMult,
            deltaTime * forwardMult,
            deltaTime * forwardMult
        });
    DirectX::XMVECTOR cameraRightDir = DirectX::XMVectorMultiply(
        m_renderer.GetCamera().GetRightVector(), {
            deltaTime * rightMult,
            deltaTime * rightMult,
            deltaTime * rightMult
        });
    DirectX::XMVECTOR cameraUpDir = DirectX::XMVectorMultiply(
        m_renderer.GetCamera().GetUpVector(), {
            deltaTime * upMult,
            deltaTime * upMult,
            deltaTime * upMult
    });

    cameraPos = DirectX::XMVectorAdd(cameraPos, cameraForwardDir);
    cameraPos = DirectX::XMVectorAdd(cameraPos, cameraRightDir);
    cameraPos = DirectX::XMVectorAdd(cameraPos, cameraUpDir);
#pragma endregion Camera_Movement

#pragma region Terrain_Generation
    if (m_window.GetKeyboard().KeyIsPressed('G')) {
        if (m_terrain_generating == false) {
            GenerateChunk();
        }
    }

    if (m_terrain_generating == true) {
        size_t work_running = m_work_items.size();
        for (size_t i = 0; i < m_work_items.size(); i++) {
            if (m_work_items[i] == NULL) {
                work_running -= 1;
                continue;
            }
            DWORD r = WaitForSingleObject(m_work_events[i], 0);
            if (r == WAIT_OBJECT_0) {
                CloseThreadpoolWork(m_work_items[i]);
                m_work_items[i] = NULL;
                work_running -= 1;
            }
        }
        if (work_running == 0)
            m_terrain_generating = false;
    }
#pragma endregion Terrain_Generation

    if (m_window.GetKeyboard().KeyIsPressed('1'))
        lod_dist = 15.0f;
    if (m_window.GetKeyboard().KeyIsPressed('2'))
        lod_dist = 100.0f;
    if (m_window.GetKeyboard().KeyIsPressed('3'))
        lod_dist = 1500.0f;

    if (m_window.GetKeyboard().KeyIsPressed('4'))
        m_renderer.GetDevice().GetContext()->RSSetState(m_renderer.solid_rs.Get());
    if (m_window.GetKeyboard().KeyIsPressed('5'))
        m_renderer.GetDevice().GetContext()->RSSetState(m_renderer.wireframe_rs.Get());
}

struct ChunkParams {
    int x, z;
    Renderer& renderer;
    HANDLE is_work_done;
};

VOID CALLBACK ChunkCallback(PTP_CALLBACK_INSTANCE, PVOID context, PTP_WORK work) {
    auto* p = static_cast<ChunkParams*>(context);
    VoxelChunk* chunk = new VoxelChunk();
    chunk->transform.Position = { static_cast<float>(p->x), -CHUNK_SIZE_Y / 2.0f, static_cast<float>(p->z) };
    chunk->Initialize(p->renderer.GetDevice());
    p->renderer.AddMesh(chunk);
    SetEvent(p->is_work_done);
    delete p;
}

void Application::GenerateChunk() {
    m_terrain_generating = true;

    SYSTEM_INFO si{};
    GetSystemInfo(&si);
    DWORD cpu_count = si.dwNumberOfProcessors;

    m_terrain_radius += 1;
    const int num_new_chunks = (m_terrain_radius * 2) * 2 + ((m_terrain_radius * 2 - 2) * 2);
    m_work_items.reserve(num_new_chunks);
    m_work_events.reserve(num_new_chunks);

    LOG_DEBUG("Generating ", num_new_chunks, " terrain chunks on ", cpu_count - 1, " threads.");

    auto SubmitChunk = [&](int x, int z) {
        HANDLE ev = CreateEvent(nullptr, TRUE, FALSE, nullptr);
        m_work_events.push_back(ev);
        auto* params = new ChunkParams{ x * CHUNK_SIZE_X, z * CHUNK_SIZE_Z, m_renderer, ev };
        PTP_WORK work = CreateThreadpoolWork(ChunkCallback, params, nullptr);
        SubmitThreadpoolWork(work);
        m_work_items.push_back(work);
    };

    for (int x = -m_terrain_radius; x < m_terrain_radius; x++) {
        SubmitChunk(x, -m_terrain_radius);
        SubmitChunk(x, (m_terrain_radius - 1));
    }

    for (int z = -m_terrain_radius + 1; z < m_terrain_radius - 1; z++) {
        SubmitChunk(-m_terrain_radius, z);
        SubmitChunk((m_terrain_radius - 1), z);
    }
}
