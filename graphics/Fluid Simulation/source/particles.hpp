/// \file particles.hpp
/// \brief Particles header file.

#pragma once

#include <glm/glm.hpp>

/// \struct Particles
struct Particles {
  int size;                 ///< Number of particles
  glm::vec3 *colors;        ///< Colors
  glm::vec3 *positions[2];  ///< Position swap buffers
  glm::vec3 *velocities[2]; ///< Velocity swap buffers
  glm::vec3 *viscosities;   ///< Viscosities
  float *densities;         ///< Densities
  float *pressures;         ///< Pressures
  float *masses;            ///< Masses
  int *cells_lookup;        ///< Grid cells lookup
  int swap;                 ///< Current swap buffer
};

/* -------------------------------------------------------------------------- */
