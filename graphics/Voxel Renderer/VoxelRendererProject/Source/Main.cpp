#include "Core/Application.h"
#include "Utilities/Logger.h"

#include <io.h>
#include <fcntl.h>

int CALLBACK WinMain(_In_ HINSTANCE hInstance, _In_opt_ HINSTANCE hPrevInstance,
    _In_ LPSTR lpCmdLine, _In_ int nCmdShow) {
    // Create console for std::cout logging
    AllocConsole();
    FILE* pFile;
    freopen_s(&pFile, "CONOUT$", "w", stdout);
    SetConsoleTitleW(L"Debug Console");

    auto& logger = Logger::getInstance();
    logger.setLevel(Logger::Level::DEBUG);

    auto app = Application(1280, 1024, L"Voxels!");
    app.Run();

    return 0;
}
