/// \file shader.cpp
/// \brief Shader implementation.

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#if defined(_WIN32) || defined(_WIN64)
#include <SDL.h>
#else
#include <SDL2/SDL.h>
#endif

#include "shader.hpp"

#ifdef USE_CUDA
// GPU functions

/// \brief Draws fluid particles using CUDA and OpenGL interop.
/// \param f Fluid simulation.
/// \param show_boundary Show boundary particles flag.
/// \param fluid_vbo Fluid VBO.
/// \param color_vbo Color VBO.
extern void Draw_GPU(Fluid &f, bool show_boundary, GLuint fluid_vbo,
  GLuint color_vbo);
#endif

/// \brief Compiles shader.
/// \param shader Shader ID.
/// \param type Shader type.
/// \param source Shader source.
/// \return True if successful, false otherwise.
bool CompileShader(GLuint &shader, GLenum type, const GLchar *source) {
  shader = glCreateShader(type);

  glShaderSource(shader, 1, &source, NULL);
  glCompileShader(shader);

  GLint compile_status;
  glGetShaderiv(shader, GL_COMPILE_STATUS, &compile_status);
  if (compile_status == GL_FALSE) {
    GLchar log[1024];
    int log_length;
    glGetShaderInfoLog(shader, 1024, &log_length, log);
    SDL_LogError(SDL_LOG_CATEGORY_APPLICATION,
      "Failed to compile shader: %s\n", log);
    glDeleteShader(shader);
    return false;
  }

  return true;
}

/// \brief Links shader program.
/// \param program Program ID.
/// \param vertex_shader Vertex shader ID.
/// \param fragment_shader Fragment shader ID.
/// \return True if successful, false otherwise.
bool LinkProgram(GLuint &program, GLuint vertex_shader,
  GLuint fragment_shader) {
  program = glCreateProgram();

  glAttachShader(program, vertex_shader);
  glAttachShader(program, fragment_shader);
  glLinkProgram(program);

  GLint link_status;
  glGetProgramiv(program, GL_LINK_STATUS, &link_status);
  if (link_status == GL_FALSE) {
    GLchar log[1024];
    int log_length;
    glGetProgramInfoLog(program, 1024, &log_length, log);
    SDL_LogError(SDL_LOG_CATEGORY_APPLICATION,
      "Failed to link program: %s\n", log);
    glDeleteProgram(program);
    return false;
  }

  return true;
}

/// \brief Creates shader program.
/// \param program Program ID.
/// \param vertex_source Vertex shader source.
/// \param fragment_source Fragment shader source.
/// \return True if successful, false otherwise.
bool CreateProgram(GLuint &program, const GLchar *vertex_source,
  const GLchar *fragment_source) {
  GLuint vertex_shader;
  if (!CompileShader(vertex_shader, GL_VERTEX_SHADER, vertex_source)) {
    SDL_LogError(SDL_LOG_CATEGORY_APPLICATION,
      "Failed to create vertex shader\n");
    return false;
  }

  GLuint fragment_shader;
  if (!CompileShader(fragment_shader, GL_FRAGMENT_SHADER, fragment_source)) {
    SDL_LogError(SDL_LOG_CATEGORY_APPLICATION,
      "Failed to create fragment shader\n");
    return false;
  }

  if (!LinkProgram(program, vertex_shader, fragment_shader)) {
    SDL_LogError(SDL_LOG_CATEGORY_APPLICATION,
      "Failed to create shader program\n");
    return false;
  }

  glDeleteShader(vertex_shader);
  glDeleteShader(fragment_shader);

  return true;
}

/* -------------------------------------------------------------------------- */

bool Shader::Create(nlohmann::json &config) {
  point_size = config["window"]["point_size"].get<float>();
  offset = config["simulation"]["space_size"].get<float>() * 0.5f;
  gpu = config["compute"]["gpu"].get<bool>();

  GLenum glew_status = glewInit();
  if (glew_status != GLEW_OK) {
    SDL_LogError(SDL_LOG_CATEGORY_APPLICATION,
      "Failed to initialize GLEW: %s\n", glewGetErrorString(glew_status));
    return false;
  }

  glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
  glEnable(GL_DEPTH_TEST);
  glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);
  glDisable(GL_CULL_FACE);
  glDisable(GL_STENCIL_TEST);

  // Bounding box shader
  const GLchar box_vertex_source[] = R"(
    #version 430 core

    layout(location = 0) in vec3 position;

    uniform mat4 mvp;

    void main() {
      gl_Position = mvp * vec4(position, 1.0);
    }
  )";

  const GLchar box_fragment_source[] = R"(
    #version 430 core

    out vec4 color;

    void main() {
      color = vec4(0.75, 0.75, 0.75, 1.0);
    }
  )";

  if (!CreateProgram(box_program, box_vertex_source, box_fragment_source)) {
    SDL_LogError(SDL_LOG_CATEGORY_APPLICATION,
      "Failed to create bounding box shader program\n");
    return false;
  }

  glGenVertexArrays(1, &box_vao);
  glBindVertexArray(box_vao);

  glGenBuffers(1, &box_vbo);
  const glm::vec3 box_vertices[] = {
    glm::vec3( offset,  offset, -offset),
    glm::vec3( offset, -offset, -offset),
    glm::vec3(-offset, -offset, -offset),
    glm::vec3(-offset,  offset, -offset),
    glm::vec3( offset,  offset,  offset),
    glm::vec3( offset, -offset,  offset),
    glm::vec3(-offset, -offset,  offset),
    glm::vec3(-offset,  offset,  offset),
  };
  glBindBuffer(GL_ARRAY_BUFFER, box_vbo);
  glBufferData(GL_ARRAY_BUFFER, sizeof(box_vertices), box_vertices,
    GL_STATIC_DRAW);

  const GLuint box_indices[] = {
    0, 1, 1, 2, 2, 3, 3, 0,
    4, 5, 5, 6, 6, 7, 7, 4,
    0, 4, 1, 5, 2, 6, 3, 7,
  };
  glGenBuffers(1, &box_ebo);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, box_ebo);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(box_indices), box_indices,
    GL_STATIC_DRAW);

  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), 0);
  glEnableVertexAttribArray(0);

  // Fluid particles shader
  const GLchar fluid_vertex_source[] = R"(
    #version 430 core

    uniform mat4 mvp;
    uniform float point_size;
    in vec3 position;
    in vec3 input_color;

    out vec3 base_color;

    void main() {
      base_color = input_color;
      gl_Position = mvp * vec4(position, 1.0);
      gl_PointSize = point_size / gl_Position.z;
    }
  )";

  const GLchar fluid_fragment_source[] = R"(
    #version 430 core

    in vec3 base_color;
    out vec4 color;

    void main() {
      float d = distance(gl_PointCoord, vec2(0.5, 0.5));
      if (d > 0.5)
        discard;
      else if (d >= 0.44)
        color = vec4(1.0, 1.0, 1.0, 1.0);
      else
        color = vec4(base_color, 1.0);
    }
  )";

  if (!CreateProgram(fluid_program, fluid_vertex_source,
    fluid_fragment_source)) {
    SDL_LogError(SDL_LOG_CATEGORY_APPLICATION,
      "Failed to create fluid particles shader program\n");
    return 1;
  }

  glGenVertexArrays(1, &fluid_vao);
  glBindVertexArray(fluid_vao);

  glGenBuffers(1, &fluid_vbo);
  glBindBuffer(GL_ARRAY_BUFFER, fluid_vbo);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), 0);
  glEnableVertexAttribArray(0);

  glGenBuffers(1, &color_vbo);
  glBindBuffer(GL_ARRAY_BUFFER, color_vbo);
  glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), 0);
  glEnableVertexAttribArray(1);

  return true;
}

void Shader::Destroy() {
  glDeleteProgram(box_program);
  glDeleteVertexArrays(1, &box_vao);
  glDeleteBuffers(1, &box_vbo);
  glDeleteBuffers(1, &box_ebo);

  glDeleteProgram(fluid_program);
  glDeleteVertexArrays(1, &fluid_vao);
  glDeleteBuffers(1, &fluid_vbo);
  glDeleteBuffers(1, &color_vbo);
}

/* -------------------------------------------------------------------------- */

void Shader::Render(int width, int height, glm::vec2 rotation, float zoom,
  bool show_boundary, Fluid &fluid) {
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glViewport(0, 0, width, height);

  float aspect = static_cast<float>(width) / height;
  glm::mat4 mv = glm::lookAt(glm::vec3(0.0f, 0.0f, zoom),
    glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
  mv = glm::rotate(mv, rotation.y, glm::vec3(1.0f, 0.0f, 0.0f));
  mv = glm::rotate(mv, rotation.x, glm::vec3(0.0f, 1.0f, 0.0f));
  glm::mat4 p = glm::perspective(45.0f, aspect, 0.01f, 100.0f);
  glm::mat4 mvp = p * mv;

  // Draw bounding box
  glUseProgram(box_program);
  glUniformMatrix4fv(glGetUniformLocation(box_program, "mvp"), 1, GL_FALSE,
    glm::value_ptr(mvp));
  glBindVertexArray(box_vao);
  glDrawElements(GL_LINES, 24, GL_UNSIGNED_INT, 0);

  mv = glm::translate(mv, glm::vec3(-offset));
  mvp = p * mv;

  // Draw fluid particles
  glUseProgram(fluid_program);
  glUniformMatrix4fv(glGetUniformLocation(fluid_program, "mvp"), 1, GL_FALSE,
    glm::value_ptr(mvp));
  glUniform1f(glGetUniformLocation(fluid_program, "point_size"),
    height / (100.0f / point_size));
  glBindVertexArray(fluid_vao);

#ifdef USE_CUDA
  if (gpu) {
    Draw_GPU(fluid, show_boundary, fluid_vbo, color_vbo);
    return;
  }
#endif

  glBindBuffer(GL_ARRAY_BUFFER, fluid_vbo);
  glBufferData(GL_ARRAY_BUFFER, fluid.fluid.size * sizeof(glm::vec3),
    fluid.fluid.positions[fluid.fluid.swap], GL_DYNAMIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, color_vbo);
  glBufferData(GL_ARRAY_BUFFER, fluid.fluid.size * sizeof(glm::vec3),
    fluid.fluid.colors, GL_DYNAMIC_DRAW);
  glDrawArrays(GL_POINTS, 0, fluid.fluid.size);

  if (show_boundary) {
    glBindBuffer(GL_ARRAY_BUFFER, fluid_vbo);
    glBufferData(GL_ARRAY_BUFFER, fluid.boundary.size * sizeof(glm::vec3),
      fluid.boundary.positions[fluid.boundary.swap], GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, color_vbo);
    glBufferData(GL_ARRAY_BUFFER, fluid.fluid.size * sizeof(glm::vec3),
      fluid.boundary.colors, GL_DYNAMIC_DRAW);
    glDrawArrays(GL_POINTS, 0, fluid.boundary.size);
  }
}

/* -------------------------------------------------------------------------- */
