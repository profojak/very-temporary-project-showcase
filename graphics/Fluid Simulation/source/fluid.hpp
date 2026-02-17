/// \file fluid.hpp
/// \brief Fluid simulation header file.

#pragma once

#include <nlohmann/json.hpp>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

#include "particles.hpp"

/// \struct Fluid
struct Fluid {
  float total_time; ///< Total physics time elapsed
  int steps;        ///< Total simulation steps taken
  float delta;      ///< Fixed simulation time step

  bool gpu;        ///< GPU compute flag
  int gpu_threads; ///< GPU threads per block
  int omp_threads; ///< OpenMP threads
  int omp_chunk;   ///< Chunk size per one OpenMP thread

#ifdef USE_CUDA
  bool gpu_init;           ///< GPU initialized flag
  cudaStream_t stream;     ///< CUDA stream
  cudaGraph_t graph[2];    ///< CUDA execution graphs
  cudaGraphExec_t exec[2]; ///< CUDA execution graph executions
#endif

  /// \brief Simulation space size.
  /// \details Simulation space size equals to the length of one side of the
  /// simulation cube.
  float space_size;
  float smooth_radius;   ///< Smoothing radius
  glm::ivec3 fluid_grid; ///< Initial grid configuration for fluid particles
  glm::ivec3 cell_grid;  ///< Grid of cells
  int cell_count;        ///< Total cell count
  /// \brief Box boundary flag.
  /// \details If false, boundary particles are created to handle the fluid
  /// particles interactions with the box edges.
  bool box_boundary;

  float gravity;   ///< Gravity constant
  float mass;      ///< Mass constant
  float density;   ///< Density constant
  float viscosity; ///< Viscosity constant
  float stiffness; ///< Stiffness constant

  bool is_init;      ///< Fluid simulation initialized flag
  int *cells_map[2]; ///< Buffer of particle to cell mapping used for sorting

  Particles fluid;    ///< Fluid particles
  Particles boundary; ///< Boundary particles

  /// \brief Creates fluid simulation.
  /// \param config Configuration JSON file.
  void Create(
    nlohmann::json &config);
  /// \brief Destroys fluid simulation.
  void Destroy();

  /// \brief Resets fluid simulation to initial state.
  void Reset();
  /// \brief Places fluid and boundary particles, if configured,
  /// in the simulation space.
  void PlaceParticles();
  /// \brief Sorts particles into cells.
  void SortParticles(
    Particles &particles);
  /// \brief Sets boundary particles mass.
  void SetBoundaryMass();

  /// \brief Advances fluid simulation by one simulation time step.
  void Update();
  /// \brief Colorizes fluid particles based on their properties.
  /// \param mode Colorization mode.
  void Colorize(
    int mode);
};

/* -------------------------------------------------------------------------- */
