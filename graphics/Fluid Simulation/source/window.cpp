/// \file window.cpp
/// \brief Window implementation.

#include "window.hpp"

bool Window::Create(nlohmann::json &config) {
  title = config["window"]["title"].get<std::string>();
  width = config["window"]["width"].get<int>();
  height = config["window"]["height"].get<int>();
  cli = config["compute"]["cli"].get<bool>();
  steps = config["compute"]["steps"].get<int>();
  box_boundary = config["simulation"]["box_boundary"].get<bool>();

  is_running = true;
  mouse_down = false;
  mouse = glm::vec2(width / 2, height / 2);
  rotation = glm::vec2(0.0f, 0.0f);
  zoom = 2.0f;

  is_simulating = false;
  show_boundary = false;
  mode = 0;

  // Initialize fluid simulation
  fluid.Create(config);
  fluid.Reset();

  // Check configuration values
  if (width <= 0 || height <= 0) {
    SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "Invalid window dimensions\n");
    return false;
  }

  if (cli) {
    if (steps <= 0) {
      SDL_LogError(SDL_LOG_CATEGORY_APPLICATION,
        "Number of simulation steps must be positive when running as CLI\n");
      return false;
    }
    return true;
  }

  // Initialize SDL
  if (SDL_Init(SDL_INIT_VIDEO) != 0) {
    SDL_LogError(SDL_LOG_CATEGORY_APPLICATION,
      "Failed to initialize SDL: %s\n", SDL_GetError());
    return false;
  }

  // Create SDL window
  window = SDL_CreateWindow(title.c_str(), SDL_WINDOWPOS_CENTERED,
    SDL_WINDOWPOS_CENTERED, width, height,
    SDL_WINDOW_SHOWN | SDL_WINDOW_OPENGL);
  if (window == NULL) {
    SDL_LogError(SDL_LOG_CATEGORY_APPLICATION,
      "Failed to create window: %s\n", SDL_GetError());
    return false;
  }

  // Create OpenGL context
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 4);
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK,
    SDL_GL_CONTEXT_PROFILE_CORE);
  SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
  SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);
  context = SDL_GL_CreateContext(window);
  if (context == NULL) {
    SDL_LogError(SDL_LOG_CATEGORY_APPLICATION,
      "Failed to create OpenGL context: %s\n", SDL_GetError());
    return false;
  }
  SDL_GL_SetSwapInterval(1);

  // Initialize shader programs
  if (!shader.Create(config)) {
    SDL_LogError(SDL_LOG_CATEGORY_APPLICATION,
      "Failed to create shader programs\n");
    return false;
  }

  return true;
}

void Window::Destroy() {
  fluid.Destroy();
  shader.Destroy();
  if (cli)
    return;

  SDL_GL_DeleteContext(context);
  SDL_DestroyWindow(window);
  SDL_Quit();
}

/* -------------------------------------------------------------------------- */

void Window::HandleEvents() {
  static SDL_Event event;
  while (SDL_PollEvent(&event)) {
    // Exit application
    if (event.type == SDL_QUIT)
      is_running = false;
    else if (event.type == SDL_KEYDOWN) {
      if (event.key.keysym.sym == SDLK_ESCAPE)
        is_running = false;

    // Fullscreen
      else if (event.key.keysym.sym == SDLK_f) {
        if (SDL_GetWindowFlags(window) & SDL_WINDOW_FULLSCREEN)
          SDL_SetWindowFullscreen(window, 0);
        else
          SDL_SetWindowFullscreen(window, SDL_WINDOW_FULLSCREEN);
      }

    // Start/stop simulation
      else if (event.key.keysym.sym == SDLK_SPACE)
        is_simulating = !is_simulating;

    // Reset simulation
      else if (event.key.keysym.sym == SDLK_r)
        fluid.Reset();

    // Show/hide boundary particles
      else if (event.key.keysym.sym == SDLK_b)
        show_boundary = !show_boundary;

    // Particles color mode
      else if (event.key.keysym.sym == SDLK_1) {
        // Base blue
        mode = 0;
        fluid.Colorize(mode);
      } else if (event.key.keysym.sym == SDLK_2) {
        // Velocity
        mode = 1;
        fluid.Colorize(mode);
      } else if (event.key.keysym.sym == SDLK_3) {
        // Pressure
        mode = 2;
        fluid.Colorize(mode);
      } else if (event.key.keysym.sym == SDLK_4) {
        // Density
        mode = 3;
        fluid.Colorize(mode);
      }

    // Camera movement
    } else if (event.type == SDL_MOUSEBUTTONDOWN) {
      if (event.button.button == SDL_BUTTON_LEFT) {
        mouse_down = true;
        mouse = glm::vec2(event.button.x, event.button.y);
      }
    } else if (event.type == SDL_MOUSEBUTTONUP) {
      if (event.button.button == SDL_BUTTON_LEFT)
        mouse_down = false;
    } else if (event.type == SDL_MOUSEMOTION) {
      int dx = event.motion.x - mouse.x;
      int dy = event.motion.y - mouse.y;
      if (mouse_down)
        rotation += glm::vec2(dx, dy) * 0.002f;
      mouse = glm::vec2(event.motion.x, event.motion.y);

    // Zoom
    } else if (event.type == SDL_MOUSEWHEEL) {
      if (event.wheel.y > 0)
        zoom /= 1.025f;
      else if (event.wheel.y < 0)
        zoom *= 1.025f;
    }
  }
}

void Window::Loop() {
  auto elapsed = SDL_GetTicks64() * 0;
  auto start = elapsed;
  auto end = elapsed;

  if (cli) {
    SDL_LogInfo(SDL_LOG_CATEGORY_APPLICATION,
      "Running simulation for %d steps\n", steps);
    is_simulating = true;
  }

  int width, height;
  while (is_running) {
    if (!cli)
      HandleEvents();

    if (is_simulating) {
      start = SDL_GetTicks64();
      fluid.Update();
      end = SDL_GetTicks64();
      elapsed += end - start;

      fluid.Colorize(mode);

      if (fluid.steps == steps) {
        is_simulating = false;
        SDL_LogInfo(SDL_LOG_CATEGORY_APPLICATION,
          "Simulation completed in %lu milliseconds\n", elapsed);
        if (cli)
          exit(0);
      }
    }

    if (!cli) {
      SDL_GetWindowSize(window, &width, &height);
      shader.Render(width, height, rotation, zoom,
        box_boundary ? false : show_boundary, fluid);
      SDL_GL_SwapWindow(window);
    }
  }
}

/* -------------------------------------------------------------------------- */
