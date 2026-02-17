/// \file shader.hpp
/// \brief Shader header file.

#pragma once

#include <GL/glew.h>
#include <glm/glm.hpp>
#include <nlohmann/json.hpp>

#include "fluid.hpp"

/// \struct Shader
struct Shader {
  GLuint box_program; ///< Box shader program
  GLuint box_vao;     ///< Box vertex array object
  GLuint box_vbo;     ///< Box vertex buffer object
  GLuint box_ebo;     ///< Box element buffer object

  GLuint fluid_program; ///< Fluid shader program
  GLuint fluid_vao;     ///< Fluid vertex array object
  GLuint fluid_vbo;     ///< Fluid vertex buffer object
  GLuint color_vbo;     ///< Fluid color buffer object

  float point_size; ///< Fluid particle screen space point size
  float offset;     ///< Translation offset
  bool gpu;         ///< GPU compute flag

  /// \brief Creates shaders.
  /// \param config Configuration JSON file.
  bool Create(
    nlohmann::json &config);
  /// \brief Destroys shaders.
  void Destroy();

  /// \brief Renders the simulation.
  /// \param width Window width.
  /// \param height Window height.
  /// \param rotation Camera rotation vector.
  /// \param zoom Camera zoom.
  /// \param show_boundary Show boundary particles flag.
  /// \param fluid Fluid simulation.
  void Render(
    int width,
    int height,
    glm::vec2 rotation,
    float zoom,
    bool show_boundary,
    Fluid &fluid);
};

/* -------------------------------------------------------------------------- */
