/// \file functions.cuh
/// \brief SPH kernel functions and colorization functions implementation.

#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

/// \brief Calculates the index of a cell in the grid.
/// \param cell The cell coordinates.
/// \param cell_grid The grid dimensions.
/// \return The index of the cell in the grid.
#ifdef USE_CUDA
__device__ __host__
#endif
inline int CellIndex(glm::ivec3 cell, glm::ivec3 cell_grid) {
  return (cell.x * cell_grid.y + cell.y) * cell_grid.z + cell.z;
}

/// \brief Calculates weight factor of a neighbor particle mass.
/// \param q The relative distance between the neighbor and the current
/// particle.
/// \param smooth_radius The smoothing radius.
/// \return The weight factor of the neighbor particle mass.
#ifdef USE_CUDA
__device__ __host__
#endif
inline float MassKernel(float q, float smooth_radius) {
  float a = 0.25f / (glm::pi<float>() * pow(smooth_radius, 3));
  return a * (q > 1.0f ? pow((2.0f - q), 3) : q * q * (3.0f * q - 6.0f) + 4.0f);
}

/// \brief Calculates weight factor of a neighbor particle viscosity.
/// \param distance The distance between the neighbor and the current particle.
/// \param smooth_radius The smoothing radius.
/// \return The weight factor of the neighbor particle viscosity.
#ifdef USE_CUDA
__device__ __host__
#endif
inline float ViscosityKernel(float distance, float smooth_radius) {
  return (smooth_radius - distance) / (glm::pi<float>() *
    pow(smooth_radius, 6));
}

/// \brief Calculates weight factor of a neighbor particle pressure.
/// \param diff The difference between the neighbor and the current particle
/// position.
/// \param q The relative distance between the neighbor and the current
/// particle.
/// \param smooth_radius The smoothing radius.
/// \return The weight factor of the neighbor particle pressure.
#ifdef USE_CUDA
__device__ __host__
#endif
inline glm::vec3 PressureKernel(glm::vec3 diff, float q, float smooth_radius) {
  glm::vec3 a = diff;
  a /= (glm::pi<float>() * (q + 1e-6f) * pow(smooth_radius, 5));
  return a * (q > 1.0f ? q * (12.0f - 3.0f * q) - 12.0f :
    (9.0f * q - 12.0f) * q);
}

/// \brief Calculates a color based on the velocity of a particle.
/// \param velocity The velocity of the particle.
/// \param diff1 The first color difference.
/// \param diff2 The second color difference.
/// \param base_color The base color.
/// \return The color based on the velocity of the particle.
#ifdef USE_CUDA
__device__ __host__
#endif
inline glm::vec3 ColorizeVelocity(glm::vec3 velocity, glm::vec3 diff1,
  glm::vec3 diff2, glm::vec3 base_color) {
  float speed = glm::length(velocity);
  if (speed < 1.0f)
    return base_color + speed * diff1;
  else if (speed < 3.0f)
    return base_color + 0.5f * (3.0f - speed) * diff1
                      + 0.5f * (speed - 1.0f) * diff2;
  else
    return base_color + diff2;
}

/// \brief Calculates a color based on the pressure of a particle.
/// \param pressure The pressure of the particle.
/// \param diff1 The first color difference.
/// \param diff2 The second color difference.
/// \param base_color The base color.
/// \return The color based on the pressure of the particle.
#ifdef USE_CUDA
__device__ __host__
#endif
inline glm::vec3 ColorizePressure(float pressure, glm::vec3 diff1,
  glm::vec3 diff2, glm::vec3 base_color) {
  if (pressure < 0.25f)
    return base_color + 4.0f * pressure * diff1;
  else if (pressure < 0.75f)
    return base_color + 2.0f * (0.75f - pressure) * diff1
                      + 2.0f * (pressure - 0.25f) * diff2;
  else
    return base_color + diff2;
}

/// \brief Calculates a color based on the density of a particle.
/// \param density The density of the particle.
/// \param diff1 The first color difference.
/// \param diff2 The second color difference.
/// \param base_color The base color.
/// \return The color based on the density of the particle.
#ifdef USE_CUDA
__device__ __host__
#endif
inline glm::vec3 ColorizeDensity(float density, glm::vec3 diff1,
  glm::vec3 diff2, glm::vec3 base_color) {
  if (density < 1.0f)
    return base_color + density * diff1;
  else if (density < 1.35f)
    return base_color + 2.0f * (1.35f - density) * diff1
                      + 2.0f * (density - 1.0f) * diff2;
  else
    return base_color + diff2;
}

/* -------------------------------------------------------------------------- */
