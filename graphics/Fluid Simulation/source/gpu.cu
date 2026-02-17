/// \file gpu.cu
/// \brief Fluid simulation on the GPU implementation.

#include <vector>

#include <cuda_runtime.h>
#if defined(_WIN32) || defined(_WIN64)
#include <SDL.h>
#else
#include <SDL2/SDL.h>
#endif
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/scan.h>

#include "fluid.hpp"
#include "functions.cuh"
#include "shader.hpp"

#include <cuda_gl_interop.h>

/// \brief Checks CUDA error.
/// \param error CUDA error.
/// \param file Current file name.
/// \param line Current line number.
void CheckError(cudaError_t error, const char* file, int line) {
  if (error != cudaSuccess)
    SDL_LogError(SDL_LOG_CATEGORY_APPLICATION,
      "CUDA error at %s:%d: %s\n", file, line, cudaGetErrorString(error));
}

/// \brief Wrapper macro to check CUDA error.
#define CHECK_ERROR(error) (CheckError(error, __FILE__, __LINE__))

/* -------------------------------------------------------------------------- */

#ifdef USE_CUDA
void Destroy_GPU(Fluid &f) {
  CHECK_ERROR(cudaFree(f.fluid.colors));
  CHECK_ERROR(cudaFree(f.fluid.positions[0]));
  CHECK_ERROR(cudaFree(f.fluid.positions[1]));
  CHECK_ERROR(cudaFree(f.fluid.velocities[0]));
  CHECK_ERROR(cudaFree(f.fluid.velocities[1]));
  CHECK_ERROR(cudaFree(f.fluid.viscosities));
  CHECK_ERROR(cudaFree(f.fluid.densities));
  CHECK_ERROR(cudaFree(f.fluid.pressures));
  CHECK_ERROR(cudaFree(f.fluid.masses));
  CHECK_ERROR(cudaFree(f.fluid.cells_lookup));

  if (!f.box_boundary) {
    CHECK_ERROR(cudaFree(f.boundary.colors));
    CHECK_ERROR(cudaFree(f.boundary.positions[0]));
    CHECK_ERROR(cudaFree(f.boundary.positions[1]));
    CHECK_ERROR(cudaFree(f.boundary.masses));
    CHECK_ERROR(cudaFree(f.boundary.cells_lookup));
  }

  f.fluid.size = 0;
  f.fluid.colors = nullptr;
  f.fluid.positions[0] = nullptr;
  f.fluid.positions[1] = nullptr;
  f.fluid.velocities[0] = nullptr;
  f.fluid.velocities[1] = nullptr;
  f.fluid.viscosities = nullptr;
  f.fluid.densities = nullptr;
  f.fluid.pressures = nullptr;
  f.fluid.masses = nullptr;
  f.fluid.cells_lookup = nullptr;

  if (!f.box_boundary) {
    f.boundary.size = 0;
    f.boundary.colors = nullptr;
    f.boundary.positions[0] = nullptr;
    f.boundary.positions[1] = nullptr;
    f.boundary.masses = nullptr;
    f.boundary.cells_lookup = nullptr;
  }

  CHECK_ERROR(cudaFree(f.cells_map[0]));
  CHECK_ERROR(cudaFree(f.cells_map[1]));

  f.cells_map[0] = nullptr;
  f.cells_map[1] = nullptr;
}

/* -------------------------------------------------------------------------- */

/// \brief Sets fluid mass on the GPU.
/// \param masses Fluid masses.
/// \param size Fluid size.
/// \param mass Fluid mass.
__global__ void SetFluidMass_CUDA(float *masses, int size, float mass) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= size)
    return;

  masses[i] = mass;
}

void PlaceParticles_GPU(Fluid &f) {
  float half_radius = 0.025f;
  glm::vec3 offset(1.0f);
  offset *= 0.5f * f.space_size;
  offset -= glm::vec3(f.fluid_grid) * 0.5f * half_radius;
  offset += glm::vec3(0.5f * half_radius);

  // Fluid particles
  std::vector<glm::vec3> positions;
  for (int i = 0; i < f.fluid_grid.x; ++i)
    for (int j = 0; j < f.fluid_grid.y; ++j)
      for (int k = 0; k < f.fluid_grid.z; ++k)
        positions.emplace_back(offset + glm::vec3(i, j, k) * half_radius);

  f.fluid.size = static_cast<int>(positions.size());

  CHECK_ERROR(cudaMalloc(&f.fluid.colors, f.fluid.size * sizeof(glm::vec3)));
  CHECK_ERROR(cudaMalloc(&f.fluid.positions[0],
    f.fluid.size * sizeof(glm::vec3)));
  CHECK_ERROR(cudaMalloc(&f.fluid.positions[1],
    f.fluid.size * sizeof(glm::vec3)));
  CHECK_ERROR(cudaMalloc(&f.fluid.velocities[0],
    f.fluid.size * sizeof(glm::vec3)));
  CHECK_ERROR(cudaMalloc(&f.fluid.velocities[1],
    f.fluid.size * sizeof(glm::vec3)));
  CHECK_ERROR(cudaMalloc(&f.fluid.viscosities,
    f.fluid.size * sizeof(glm::vec3)));
  CHECK_ERROR(cudaMalloc(&f.fluid.densities, f.fluid.size * sizeof(float)));
  CHECK_ERROR(cudaMalloc(&f.fluid.pressures, f.fluid.size * sizeof(float)));
  CHECK_ERROR(cudaMalloc(&f.fluid.masses, f.fluid.size * sizeof(float)));
  CHECK_ERROR(cudaMalloc(&f.fluid.cells_lookup,
    (f.cell_count + 1) * sizeof(int)));

  CHECK_ERROR(cudaMemcpy(f.fluid.positions[0], positions.data(),
    f.fluid.size * sizeof(glm::vec3), cudaMemcpyHostToDevice));
  CHECK_ERROR(cudaMemset(f.fluid.velocities[0], 0,
    f.fluid.size * sizeof(glm::vec3)));
  CHECK_ERROR(cudaMemset(f.fluid.viscosities, 0,
    f.fluid.size * sizeof(glm::vec3)));
  CHECK_ERROR(cudaMemset(f.fluid.densities, 0, f.fluid.size * sizeof(float)));
  CHECK_ERROR(cudaMemset(f.fluid.pressures, 0, f.fluid.size * sizeof(float)));

  int blocks = (f.fluid.size + f.gpu_threads - 1) / f.gpu_threads;
  SetFluidMass_CUDA<<<blocks, f.gpu_threads>>>(
    f.fluid.masses, f.fluid.size, f.mass);
  CHECK_ERROR(cudaGetLastError());

  CHECK_ERROR(cudaMalloc(&f.cells_map[1], (f.cell_count + 1) * sizeof(int)));

  if (f.box_boundary) {
    CHECK_ERROR(cudaMalloc(&f.cells_map[0], f.fluid.size * sizeof(int)));
    CHECK_ERROR(cudaMemset(f.cells_map[0], 0, f.fluid.size * sizeof(int)));
    return;
  }

  // Boundary particles
  positions.clear();
  glm::vec3 normalized = 2.0f * glm::vec3(f.cell_grid) - glm::vec3(1.0f);
  normalized = 1.0f / normalized * f.space_size;

  for (int x = 0; x < 2 * f.cell_grid.x; x++) {
    for (int y = 0; y < 2 * f.cell_grid.y; y++) {
      positions.emplace_back(0.999f * glm::vec3(x, y, 0) * normalized);
      positions.emplace_back(0.999f * glm::vec3(x, y, 2 * f.cell_grid.z - 1) *
        normalized);
    }
  }

  for (int x = 0; x < 2 * f.cell_grid.x; x++) {
    for (int z = 0; z < 2 * f.cell_grid.z - 2; z++) {
      positions.emplace_back(0.999f * glm::vec3(x, 0, z + 1) * normalized);
      positions.emplace_back(0.999f *
        glm::vec3(x, 2 * f.cell_grid.y - 1, z + 1) * normalized);
    }
  }

  for (int y = 0; y < 2 * f.cell_grid.y - 2; y++) {
    for (int z = 0; z < 2 * f.cell_grid.z - 2; z++) {
      positions.emplace_back(0.999f * glm::vec3(0, y + 1, z + 1) * normalized);
      positions.emplace_back(0.999f *
        glm::vec3(2 * f.cell_grid.x - 1, y + 1, z + 1) * normalized);
    }
  }

  f.boundary.size = static_cast<int>(positions.size());

  CHECK_ERROR(cudaMalloc(&f.boundary.colors,
    f.boundary.size * sizeof(glm::vec3)));
  CHECK_ERROR(cudaMalloc(&f.boundary.positions[0],
    f.boundary.size * sizeof(glm::vec3)));
  CHECK_ERROR(cudaMalloc(&f.boundary.positions[1],
    f.boundary.size * sizeof(glm::vec3)));
  CHECK_ERROR(cudaMalloc(&f.boundary.masses,
    f.boundary.size * sizeof(float)));
  CHECK_ERROR(cudaMalloc(&f.boundary.cells_lookup,
    (f.cell_count + 1) * sizeof(int)));

  CHECK_ERROR(cudaMemcpy(f.boundary.positions[0], positions.data(),
    f.boundary.size * sizeof(glm::vec3), cudaMemcpyHostToDevice));
  CHECK_ERROR(cudaMemset(f.boundary.masses, 0,
    f.boundary.size * sizeof(float)));

  CHECK_ERROR(cudaMalloc(&f.cells_map[0],
    glm::max(f.fluid.size, f.boundary.size) * sizeof(int)));
  CHECK_ERROR(cudaMemset(f.cells_map[0],
    0, glm::max(f.fluid.size, f.boundary.size) * sizeof(int)));
}

/* -------------------------------------------------------------------------- */

/// \brief Maps particles to cells on the GPU.
/// \param positions Particle positions.
/// \param size Particle size.
/// \param cells_map Cells map.
/// \param smooth_radius Smooth radius.
/// \param cell_grid Cell grid.
__global__ void MapParticlesToCells_CUDA(glm::vec3 *positions, int size,
  int *cells_map, float smooth_radius, glm::ivec3 cell_grid) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= size)
    return;

  cells_map[i] = CellIndex(positions[i] / smooth_radius, cell_grid);
}

__global__ void InitializeCellsLookup_CUDA(int *cells_map, int size,
  int *cells_lookup) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= size)
    return;

  atomicAdd(&cells_lookup[cells_map[i]], 1);
}

/// \brief Sorts particles by cell index on the GPU.
/// \param current_positions Current particle positions.
/// \param next_positions Next particle positions.
/// \param current_velocities Current particle velocities.
/// \param next_velocities Next particle velocities.
/// \param size Particle size.
/// \param cells_map Cells map.
/// \param cells_counter Cells counter.
__global__ void SortParticles_CUDA(glm::vec3 *current_positions,
  glm::vec3 *next_positions, glm::vec3 *current_velocities,
  glm::vec3 *next_velocities, int size, int *cells_map, int *cells_counter) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= size)
    return;

  int index = cells_map[i];
  int offset = atomicSub(&cells_counter[index + 1], 1);
  offset--;

  glm::vec3 position = current_positions[i];
  next_positions[offset] = position;
  if (current_velocities != nullptr) {
    glm::vec3 velocity = current_velocities[i];
    next_velocities[offset] = velocity;
  }
}

void SortParticles_GPU(Particles &particles, Fluid &f) {
  // Map particles to cells
  int blocks = (particles.size + f.gpu_threads - 1) / f.gpu_threads;
  MapParticlesToCells_CUDA<<<blocks, f.gpu_threads>>>(
    particles.positions[particles.swap], particles.size, f.cells_map[0],
    f.smooth_radius, f.cell_grid);
  CHECK_ERROR(cudaGetLastError());

  // Initialize cells lookup
  CHECK_ERROR(cudaMemset(particles.cells_lookup, 0,
    (f.cell_count + 1) * sizeof(int)));
  blocks = (f.cell_count + f.gpu_threads - 1) / f.gpu_threads;
  InitializeCellsLookup_CUDA<<<blocks, f.gpu_threads>>>(
    f.cells_map[0], particles.size, particles.cells_lookup);
  CHECK_ERROR(cudaGetLastError());

  thrust::device_ptr<int> cells_lookup(particles.cells_lookup);
  thrust::exclusive_scan(cells_lookup, cells_lookup + f.cell_count + 1,
    cells_lookup);

  // Sort particles by cell index
  CHECK_ERROR(cudaMemcpy(f.cells_map[1], particles.cells_lookup,
    (f.cell_count + 1) * sizeof(int), cudaMemcpyDeviceToDevice));

  int next_swap = glm::abs(particles.swap - 1);
  blocks = (particles.size + f.gpu_threads - 1) / f.gpu_threads;
  SortParticles_CUDA<<<blocks, f.gpu_threads>>>(
    particles.positions[particles.swap], particles.positions[next_swap],
    particles.velocities[particles.swap], particles.velocities[next_swap],
    particles.size, f.cells_map[0], f.cells_map[1]);
  CHECK_ERROR(cudaGetLastError());

  particles.swap = next_swap;
}

/* -------------------------------------------------------------------------- */

/// \brief Sets boundary mass on the GPU.
/// \param positions Boundary positions.
/// \param masses Boundary masses.
/// \param size Boundary size.
/// \param density Fluid density.
/// \param smooth_radius Smooth radius.
/// \param cell_count Cell count.
/// \param cell_grid Cell grid.
/// \param cells_lookup Cells lookup.
__global__ void SetBoundaryMass_CUDA(glm::vec3 *positions, float *masses,
  int size, float density, float smooth_radius, int cell_count,
  glm::ivec3 cell_grid, int *cells_lookup) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= size)
    return;

  glm::ivec3 cell(positions[i] / smooth_radius);

  for (int x = -1; x <= 1; x++) {
    for (int y = -1; y <= 1; y++) {
      for (int z = -1; z <= 1; z++) {
        int index = CellIndex(cell + glm::ivec3(x, y, z), cell_grid);
        if (index < 0 || index >= cell_count)
          continue;

        float mass_contribution = 0.0f;
        for (int j = cells_lookup[index]; j < cells_lookup[index + 1]; j++) {

          float distance = glm::length(positions[i] - positions[j]);
          float q = 2.0f * distance / smooth_radius;
          if (q > 2.0f || q < 1e-6f)
            continue;
          mass_contribution += MassKernel(q, smooth_radius);
        }
        masses[i] += mass_contribution;
      }
    }
  }

  masses[i] = 1.4f * density / glm::max(1e-6f, masses[i]);
}

void SetBoundaryMass_GPU(Fluid &f) {
  int blocks = (f.boundary.size + f.gpu_threads - 1) / f.gpu_threads;
  SetBoundaryMass_CUDA<<<blocks, f.gpu_threads>>>(
    f.boundary.positions[f.boundary.swap], f.boundary.masses, f.boundary.size,
    f.density, f.smooth_radius, f.cell_count, f.cell_grid,
    f.boundary.cells_lookup);
  CHECK_ERROR(cudaGetLastError());
}

/* -------------------------------------------------------------------------- */

/// \brief Applies gravity on the GPU.
/// \param velocities Particle velocities.
/// \param size Particle size.
/// \param gravity Gravity.
/// \param delta Time delta.
__global__ void ApplyGravity_CUDA(glm::vec3 *velocities, int size,
  float gravity, float delta) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= size)
    return;

  velocities[i].y += gravity * delta;
}

/// \brief Applies viscosity on the GPU.
/// \param positions Particle positions.
/// \param velocities Particle velocities.
/// \param viscosities Particle viscosities.
/// \param masses Particle masses.
/// \param size Particle size.
/// \param viscosity Viscosity.
/// \param density Fluid density.
/// \param smooth_radius Smooth radius.
/// \param cell_count Cell count.
/// \param cell_grid Cell grid.
/// \param cells_lookup Cells lookup.
__global__ void ApplyViscosity_CUDA(glm::vec3 *positions,
  glm::vec3 *velocities, glm::vec3 *viscosities, float *masses, int size,
  float viscosity, float density, float smooth_radius, int cell_count,
  glm::ivec3 cell_grid, int *cells_lookup) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= size)
    return;

  glm::ivec3 cell(positions[i] / smooth_radius);

  viscosities[i] = glm::vec3(0.0f);
  for (int x = -1; x <= 1; x++) {
    for (int y = -1; y <= 1; y++) {
      for (int z = -1; z <= 1; z++) {
        int index = CellIndex(cell + glm::ivec3(x, y, z), cell_grid);
        if (index < 0 || index >= cell_count)
          continue;

        glm::vec3 viscosity_contribution(0.0f);
        for (int j = cells_lookup[index]; j < cells_lookup[index + 1]; j++) {

          float distance = glm::length(positions[i] - positions[j]);
          if (distance > smooth_radius)
            continue;
          viscosity_contribution += masses[j] *
            ((velocities[j] - velocities[i]) / density) *
            ViscosityKernel(distance, smooth_radius);
        }
        viscosities[i] += viscosity_contribution;
      }
    }
  }
  viscosities[i] *= viscosity;
}

/// \brief Adds viscosity to velocity on the GPU.
/// \param velocities Particle velocities.
/// \param viscosities Particle viscosities.
/// \param size Particle size.
/// \param delta Time delta.
__global__ void AddViscosity_CUDA(glm::vec3 *velocities, glm::vec3 *viscosities,
  int size, float delta) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= size)
    return;

  velocities[i] += viscosities[i] * delta;
}

/// \brief Applies density on the GPU.
/// \param fluid_positions Fluid particle positions.
/// \param fluid_densities Fluid particle densities.
/// \param fluid_masses Fluid particle masses.
/// \param size Fluid particle size.
/// \param boundary_positions Boundary particle positions.
/// \param boundary_masses Boundary particle masses.
/// \param density Fluid density.
/// \param box_boundary Box boundary.
/// \param smooth_radius Smooth radius.
/// \param cell_count Cell count.
/// \param cell_grid Cell grid.
/// \param fluid_cells_lookup Fluid cells lookup.
/// \param boundary_cells_lookup Boundary cells lookup.
__global__ void ApplyDensity_CUDA(glm::vec3 *fluid_positions,
  float *fluid_densities, float *fluid_masses, int size,
  glm::vec3 *boundary_positions, float *boundary_masses, float density,
  bool box_boundary, float smooth_radius, int cell_count, glm::ivec3 cell_grid,
  int *fluid_cells_lookup, int *boundary_cells_lookup) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= size)
    return;

  glm::ivec3 cell(fluid_positions[i] / smooth_radius);

  for (int x = -1; x <= 1; x++) {
    for (int y = -1; y <= 1; y++) {
      for (int z = -1; z <= 1; z++) {
        int index = CellIndex(cell + glm::ivec3(x, y, z), cell_grid);
        if (index < 0 || index >= cell_count)
          continue;

        // Fluid particles
        float density_contribution = 0.0f;
        for (int j = fluid_cells_lookup[index];
          j < fluid_cells_lookup[index + 1]; j++) {

          float distance = glm::length(fluid_positions[i] - fluid_positions[j]);
          float q = 2.0f * fabs(distance) / smooth_radius;
          if (q > 2.0f || q < 1e-6f)
            continue;
          density_contribution += fluid_masses[j] *
            MassKernel(q, smooth_radius);
        }

        // Boundary particles
        if (!box_boundary) {
          for (int j = boundary_cells_lookup[index];
            j < boundary_cells_lookup[index + 1]; j++) {

            float distance = glm::length(fluid_positions[i] -
              boundary_positions[j]);
            float q = 2.0f * fabs(distance) / smooth_radius;
            if (q > 2.0f || q < 1e-6f)
              continue;
            density_contribution += boundary_masses[j] *
              MassKernel(q, smooth_radius);
          }
        }

        fluid_densities[i] += density_contribution;
      }
    }
  }
}

/// \brief Sets pressure on the GPU.
/// \param pressures Fluid particle pressures.
/// \param densities Fluid particle densities.
/// \param size Fluid particle size.
/// \param stiffness Stiffness.
/// \param density Fluid density.
__global__ void SetPressure_CUDA(float *pressures, float *densities, int size,
  float stiffness, float density) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= size)
    return;

  pressures[i] = glm::max(0.0f, stiffness *
    (powf(densities[i] / density, 7.0f) - 1.0f));
}

/// \brief Applies pressure on the GPU.
/// \param fluid_positions Fluid particle positions.
/// \param fluid_velocities Fluid particle velocities.
/// \param fluid_masses Fluid particle masses.
/// \param fluid_densities Fluid particle densities.
/// \param fluid_pressures Fluid particle pressures.
/// \param size Fluid particle size.
/// \param boundary_positions Boundary particle positions.
/// \param boundary_masses Boundary particle masses.
/// \param delta Time delta.
/// \param box_boundary Box boundary.
/// \param smooth_radius Smooth radius.
/// \param cell_count Cell count.
/// \param cell_grid Cell grid.
/// \param fluid_cells_lookup Fluid cells lookup.
/// \param boundary_cells_lookup Boundary cells lookup.
__global__ void ApplyPressure_CUDA(glm::vec3 *fluid_positions,
  glm::vec3 *fluid_velocities, float *fluid_masses, float *fluid_densities,
  float *fluid_pressures, int size, glm::vec3 *boundary_positions,
  float *boundary_masses, float delta, bool box_boundary, float smooth_radius,
  int cell_count, glm::ivec3 cell_grid, int *fluid_cells_lookup,
  int *boundary_cells_lookup) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= size)
    return;

  glm::ivec3 cell(fluid_positions[i] / smooth_radius);

  glm::vec3 velocity(0.0f);
  for (int x = -1; x <= 1; x++) {
    for (int y = -1; y <= 1; y++) {
      for (int z = -1; z <= 1; z++) {
        int index = CellIndex(cell + glm::ivec3(x, y, z), cell_grid);
        if (index < 0 || index >= cell_count)
          continue;

        // Fluid particles
        glm::vec3 velocity_contribution(0.0f);
        for (int j = fluid_cells_lookup[index];
          j < fluid_cells_lookup[index + 1]; j++) {

          if (i == j)
            continue;
          glm::vec3 diff = fluid_positions[i] - fluid_positions[j];
          float q = 2.0f * fabs(glm::length(diff)) / smooth_radius;
          if (q > 2.0f)
            continue;
          velocity_contribution += -fluid_masses[j] *
            (fluid_pressures[i] /
            glm::max(1e-6f, fluid_densities[i] * fluid_densities[i]) +
            fluid_pressures[j] /
            glm::max(1e-6f, fluid_densities[j] * fluid_densities[j])) *
            PressureKernel(diff, q, smooth_radius);
        }

        // Boundary particles
        if (!box_boundary) {
          for (int j = boundary_cells_lookup[index];
            j < boundary_cells_lookup[index + 1]; j++) {

            glm::vec3 diff = fluid_positions[i] - boundary_positions[j];
            float q = 2.0f * fabs(glm::length(diff)) / smooth_radius;
            if (q > 2.0f)
              continue;
            velocity_contribution += -boundary_masses[j] *
              (fluid_pressures[i] /
              glm::max(1e-6f, fluid_densities[i] * fluid_densities[i])) *
              PressureKernel(diff, q, smooth_radius);
          }
        }

        velocity += velocity_contribution;
      }
    }
  }

  if (glm::length(velocity) > 1000.0f)
    velocity = glm::normalize(velocity) * 1000.0f;
  fluid_velocities[i] += velocity * delta;
}

/// \brief Integrates particles on the GPU.
/// \param positions Particle positions.
/// \param velocities Particle velocities.
/// \param size Particle size.
/// \param delta Time delta.
/// \param space_size Space size.
__global__ void Integrate_CUDA(glm::vec3 *positions, glm::vec3 *velocities,
  int size, float delta, float space_size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= size)
    return;

  positions[i] += velocities[i] * delta;

  float min_bound = space_size - 0.999f * space_size;
  float max_bound = 0.999f * space_size;

  // Check box boundaries
  if (positions[i].x < min_bound) {
    positions[i].x = min_bound;
    velocities[i].x = -velocities[i].x;
  } else if (positions[i].x > max_bound) {
    positions[i].x = max_bound;
    velocities[i].x = -velocities[i].x;
  }
  if (positions[i].y < min_bound) {
    positions[i].y = min_bound;
    velocities[i].y = -velocities[i].y;
  } else if (positions[i].y > max_bound) {
    positions[i].y = max_bound;
    velocities[i].y = -velocities[i].y;
  }
  if (positions[i].z < min_bound) {
    positions[i].z = min_bound;
    velocities[i].z = -velocities[i].z;
  } else if (positions[i].z > max_bound) {
    positions[i].z = max_bound;
    velocities[i].z = -velocities[i].z;
  }
}

void Update_GPU(Fluid &f) {
  CHECK_ERROR(cudaGraphLaunch(f.exec[f.fluid.swap], f.stream));
  CHECK_ERROR(cudaStreamSynchronize(f.stream));
}

/* -------------------------------------------------------------------------- */

/// \brief Colorizes particles on the GPU.
/// \param colors Particle colors.
/// \param velocities Particle velocities.
/// \param pressures Particle pressures.
/// \param densities Particle densities.
/// \param size Particle size.
/// \param mode Colorization mode.
/// \param base_color Base color.
__global__ void Colorize_CUDA(glm::vec3 *colors, glm::vec3 *velocities,
  float *pressures, float *densities, int size, int mode,
  glm::vec3 base_color) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= size)
    return;

  // Base color
  if (mode == 0)
    colors[i] = base_color;

  // Velocity
  else if (mode == 1) {
    glm::vec3 diff1(0.5f, 0.3f, 0.0f);
    glm::vec3 diff2(0.7f, -0.3f, -0.7f);
    colors[i] = ColorizeVelocity(velocities[i], diff1, diff2, base_color);
  }

  // Pressure
  else if (mode == 2) {
    glm::vec3 diff1(0.5f, 0.3f, 0.0f);
    glm::vec3 diff2(0.7f, -0.3f, -0.7f);
    colors[i] = ColorizePressure(pressures[i], diff1, diff2, base_color);
  }

  // Density
  else if (mode == 3) {
    glm::vec3 diff1(0.5f, 0.3f, 0.0f);
    glm::vec3 diff2(0.7f, -0.3f, -0.7f);
    colors[i] = ColorizeDensity(densities[i], diff1, diff2, base_color);
  }
}

void Colorize_GPU(int mode, glm::vec3 base_color, Fluid &f) {
  int blocks = (f.fluid.size + f.gpu_threads - 1) / f.gpu_threads;
  Colorize_CUDA<<<blocks, f.gpu_threads>>>(
    f.fluid.colors, f.fluid.velocities[f.fluid.swap],
    f.fluid.pressures, f.fluid.densities, f.fluid.size, mode, base_color);
  CHECK_ERROR(cudaGetLastError());
}

void Draw_GPU(Fluid &f, bool show_boundary, GLuint fluid_vbo,
  GLuint color_vbo) {
  // Fluid positions
  struct cudaGraphicsResource *cuda_fluid_resource = nullptr;

  glBindBuffer(GL_ARRAY_BUFFER, fluid_vbo);
  glBufferData(GL_ARRAY_BUFFER, f.fluid.size * sizeof(glm::vec3),
    NULL, GL_DYNAMIC_DRAW);

  CHECK_ERROR(cudaGraphicsGLRegisterBuffer(&cuda_fluid_resource, fluid_vbo,
    cudaGraphicsMapFlagsWriteDiscard));
  CHECK_ERROR(cudaGraphicsMapResources(1, &cuda_fluid_resource));

  void *positions = nullptr;
  size_t bytes = 0;
  CHECK_ERROR(cudaGraphicsResourceGetMappedPointer(&positions, &bytes,
    cuda_fluid_resource));

  CHECK_ERROR(cudaMemcpy(positions, f.fluid.positions[f.fluid.swap],
    f.fluid.size * sizeof(glm::vec3), cudaMemcpyDeviceToDevice));

  // Fluid colors
  struct cudaGraphicsResource *cuda_color_resource = nullptr;

  glBindBuffer(GL_ARRAY_BUFFER, color_vbo);
  glBufferData(GL_ARRAY_BUFFER, f.fluid.size * sizeof(glm::vec3),
    NULL, GL_DYNAMIC_DRAW);

  CHECK_ERROR(cudaGraphicsGLRegisterBuffer(&cuda_color_resource, color_vbo,
    cudaGraphicsMapFlagsWriteDiscard));
  CHECK_ERROR(cudaGraphicsMapResources(1, &cuda_color_resource));

  void *colors = nullptr;
  CHECK_ERROR(cudaGraphicsResourceGetMappedPointer(&colors, &bytes,
    cuda_color_resource));

  CHECK_ERROR(cudaMemcpy(colors, f.fluid.colors,
    f.fluid.size * sizeof(glm::vec3), cudaMemcpyDeviceToDevice));

  // Draw fluid
  glDrawArrays(GL_POINTS, 0, f.fluid.size);

  CHECK_ERROR(cudaGraphicsUnmapResources(1, &cuda_fluid_resource));
  CHECK_ERROR(cudaGraphicsUnmapResources(1, &cuda_color_resource));

  if (!show_boundary)
    return;

  // Boundary positions
  glBindBuffer(GL_ARRAY_BUFFER, fluid_vbo);
  glBufferData(GL_ARRAY_BUFFER, f.boundary.size * sizeof(glm::vec3),
    NULL, GL_DYNAMIC_DRAW);

  CHECK_ERROR(cudaGraphicsGLRegisterBuffer(&cuda_fluid_resource, fluid_vbo,
    cudaGraphicsMapFlagsWriteDiscard));
  CHECK_ERROR(cudaGraphicsMapResources(1, &cuda_fluid_resource));

  CHECK_ERROR(cudaGraphicsResourceGetMappedPointer(&positions, &bytes,
    cuda_fluid_resource));

  CHECK_ERROR(cudaMemcpy(positions, f.boundary.positions[f.boundary.swap],
    f.boundary.size * sizeof(glm::vec3), cudaMemcpyDeviceToDevice));

  // Boundary colors
  glBindBuffer(GL_ARRAY_BUFFER, color_vbo);
  glBufferData(GL_ARRAY_BUFFER, f.boundary.size * sizeof(glm::vec3),
    NULL, GL_DYNAMIC_DRAW);

  CHECK_ERROR(cudaGraphicsGLRegisterBuffer(&cuda_color_resource, color_vbo,
    cudaGraphicsMapFlagsWriteDiscard));
  CHECK_ERROR(cudaGraphicsMapResources(1, &cuda_color_resource));

  CHECK_ERROR(cudaGraphicsResourceGetMappedPointer(&colors, &bytes,
    cuda_color_resource));

  CHECK_ERROR(cudaMemcpy(colors, f.boundary.colors,
    f.boundary.size * sizeof(glm::vec3), cudaMemcpyDeviceToDevice));

  // Draw boundary
  glDrawArrays(GL_POINTS, 0, f.boundary.size);

  CHECK_ERROR(cudaGraphicsUnmapResources(1, &cuda_fluid_resource));
  CHECK_ERROR(cudaGraphicsUnmapResources(1, &cuda_color_resource));
}

/* -------------------------------------------------------------------------- */

void Setup_GPU(Fluid &f) {
  int blocks = (f.fluid.size + f.gpu_threads - 1) / f.gpu_threads;
  CHECK_ERROR(cudaStreamCreate(&f.stream));

  // Setup for each of the swap buffers
  for (int i = 0; i < 2; i++) {
    CHECK_ERROR(cudaStreamBeginCapture(f.stream, cudaStreamCaptureModeGlobal));

    ApplyGravity_CUDA<<<blocks, f.gpu_threads, 0, f.stream>>>(
      f.fluid.velocities[i], f.fluid.size, f.gravity, f.delta);
    CHECK_ERROR(cudaGetLastError());

    ApplyViscosity_CUDA<<<blocks, f.gpu_threads, 0, f.stream>>>(
      f.fluid.positions[i], f.fluid.velocities[i], f.fluid.viscosities,
      f.fluid.masses, f.fluid.size, f.viscosity, f.density, f.smooth_radius,
      f.cell_count, f.cell_grid, f.fluid.cells_lookup);
    CHECK_ERROR(cudaGetLastError());

    AddViscosity_CUDA<<<blocks, f.gpu_threads, 0, f.stream>>>(
      f.fluid.velocities[i], f.fluid.viscosities, f.fluid.size, f.delta);
    CHECK_ERROR(cudaGetLastError());

    CHECK_ERROR(cudaMemsetAsync(f.fluid.densities, 0,
      f.fluid.size * sizeof(float), f.stream));

    ApplyDensity_CUDA<<<blocks, f.gpu_threads, 0, f.stream>>>(
      f.fluid.positions[i], f.fluid.densities, f.fluid.masses, f.fluid.size,
      f.boundary.positions[f.boundary.swap], f.boundary.masses, f.density,
      f.box_boundary, f.smooth_radius, f.cell_count, f.cell_grid,
      f.fluid.cells_lookup, f.boundary.cells_lookup);
    CHECK_ERROR(cudaGetLastError());

    SetPressure_CUDA<<<blocks, f.gpu_threads, 0, f.stream>>>(
      f.fluid.pressures, f.fluid.densities, f.fluid.size, f.stiffness,
      f.density);
    CHECK_ERROR(cudaGetLastError());

    ApplyPressure_CUDA<<<blocks, f.gpu_threads, 0, f.stream>>>(
      f.fluid.positions[i], f.fluid.velocities[i], f.fluid.masses,
      f.fluid.densities, f.fluid.pressures, f.fluid.size,
      f.boundary.positions[f.boundary.swap], f.boundary.masses, f.delta,
      f.box_boundary, f.smooth_radius, f.cell_count, f.cell_grid,
      f.fluid.cells_lookup, f.boundary.cells_lookup);
    CHECK_ERROR(cudaGetLastError());

    Integrate_CUDA<<<blocks, f.gpu_threads, 0, f.stream>>>(
      f.fluid.positions[i], f.fluid.velocities[i], f.fluid.size, f.delta,
      f.space_size);
    CHECK_ERROR(cudaGetLastError());

    CHECK_ERROR(cudaStreamEndCapture(f.stream, &f.graph[i]));
    CHECK_ERROR(cudaGraphInstantiate(&f.exec[i], f.graph[i], NULL, NULL, 0));
  }

  f.gpu_init = true;
}
#endif

/* -------------------------------------------------------------------------- */
