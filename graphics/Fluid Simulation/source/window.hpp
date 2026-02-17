/// \file window.hpp
/// \brief Window header file.

#pragma once

#include <glm/glm.hpp>
#include <nlohmann/json.hpp>
#if defined(_WIN32) || defined(_WIN64)
#include <SDL.h>
#else
#include <SDL2/SDL.h>
#endif

#include <string>

#include "fluid.hpp"
#include "shader.hpp"

/// \struct Window
struct Window {
  /// \brief Pointer to SDL2 window object.
  SDL_Window *window;
  /// \brief OpenGL context.
  SDL_GLContext context;

  /// \brief Fluid object.
  /// \details Contains all the fluid particles and handles the simulation.
  Fluid fluid;
  /// \brief Shader object.
  /// \details Contains the shader program to render the fluid particles.
  Shader shader;

  std::string title; ///< Window title
  int width;         ///< Width
  int height;        ///< Height
  /// \brief CLI flag.
  /// \details If true, the window will not be created and the simulation will
  /// be run in the command line. It only works if `steps` in the configuration
  /// file is positive.
  bool cli;

  /// \brief Running flag.
  /// \details If true, the window loop is running. On false, the loop exits.
  bool is_running;
  bool mouse_down;    ///< Mouse down flag
  glm::vec2 mouse;    ///< Mouse position
  /// \brief Camera rotation.
  /// \details The vector is computed from the mouse movement. It rotates
  /// the viewport.
  glm::vec2 rotation;
  float zoom;         ///< Camera zoom

  /// \brief Simulating flag.
  /// \details If true, the simulation is running.
  bool is_simulating;
  bool show_boundary; ///< Show boundary particles flag
  /// \brief Box boundary flag.
  /// \details If false, boundary particles are created to handle the fluid
  /// particles interactions with the box edges.
  bool box_boundary;
  /// \brief Particle color mode.
  /// \details 0: blue, 1: velocity, 2: pressure, 3: density.
  int mode;
  int steps;          ///< Number of simulation steps to benchmark

  /// \brief Creates window.
  /// \param config Configuration JSON file.
  /// \return True if the window was created successfully.
  bool Create(
    nlohmann::json &config);
  /// \brief Destroys window.
  void Destroy();

  /// \brief Handles input events.
  void HandleEvents();
  /// \brief Loop function.
  /// \details Handles input events, updates the simulation and renders it.
  void Loop();
};

/* -------------------------------------------------------------------------- */
