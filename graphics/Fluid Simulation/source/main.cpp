/// \file main.cpp
/// \brief File with the main function.

#include <nlohmann/json.hpp>
#if defined(_WIN32) || defined(_WIN64)
#include <SDL.h>
#else
#include <SDL2/SDL.h>
#endif

#include <fstream>

#include "window.hpp"

/// \brief Main function.
/// \param argc Number of arguments.
/// \param argv Arguments.
int main(int argc, char* argv[]) {
  nlohmann::json json;

  if (argc == 1) {
    std::ifstream file("config.json");
    if (file.fail()) {
      SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "Failed to open JSON file\n");
      exit(1);
    }
    json = nlohmann::json::parse(file);
  } else if (argc > 1) {
    std::ifstream file(argv[1]);
    if (file.fail()) {
      SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "Failed to open JSON file\n");
      exit(1);
    }
    json = nlohmann::json::parse(file);
  }

  Window window;
  if (!window.Create(json)) {
    SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "Failed to create simulation\n");
    exit(1);
  }

  window.Loop();
  window.Destroy();

  return 0;
}

/* -------------------------------------------------------------------------- */
