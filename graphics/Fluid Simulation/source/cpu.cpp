/// \file cpu.cpp
/// \brief Fluid simulation on CPU implementation.

#include <numeric>
#include <vector>

#include <omp.h>

#include "fluid.hpp"
#include "functions.cuh"

void Destroy_CPU(Fluid &f) {
  delete[] f.fluid.colors;
  delete[] f.fluid.positions[0];
  delete[] f.fluid.positions[1];
  delete[] f.fluid.velocities[0];
  delete[] f.fluid.velocities[1];
  delete[] f.fluid.viscosities;
  delete[] f.fluid.densities;
  delete[] f.fluid.pressures;
  delete[] f.fluid.masses;
  delete[] f.fluid.cells_lookup;

  if (!f.box_boundary) {
    delete[] f.boundary.colors;
    delete[] f.boundary.positions[0];
    delete[] f.boundary.positions[1];
    delete[] f.boundary.masses;
    delete[] f.boundary.cells_lookup;
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

  delete[] f.cells_map[0];
  delete[] f.cells_map[1];

  f.cells_map[0] = nullptr;
  f.cells_map[1] = nullptr;
}

/* -------------------------------------------------------------------------- */

void PlaceParticles_CPU(Fluid &f) {
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
  f.omp_chunk = f.fluid.size / f.omp_threads;

  f.fluid.colors = new glm::vec3[f.fluid.size];
  f.fluid.positions[0] = new glm::vec3[f.fluid.size];
  f.fluid.positions[1] = new glm::vec3[f.fluid.size];
  f.fluid.velocities[0] = new glm::vec3[f.fluid.size];
  f.fluid.velocities[1] = new glm::vec3[f.fluid.size];
  f.fluid.viscosities = new glm::vec3[f.fluid.size];
  f.fluid.densities = new float[f.fluid.size];
  f.fluid.pressures = new float[f.fluid.size];
  f.fluid.masses = new float[f.fluid.size];
  f.fluid.cells_lookup = new int[f.cell_count + 1];

  std::copy(positions.begin(), positions.end(), f.fluid.positions[0]);
  std::fill(f.fluid.velocities[0], f.fluid.velocities[0] + f.fluid.size,
    glm::vec3(0.0f));
  std::fill(f.fluid.viscosities, f.fluid.viscosities + f.fluid.size,
    glm::vec3(0.0f));
  std::fill(f.fluid.densities, f.fluid.densities + f.fluid.size, 0.0f);
  std::fill(f.fluid.pressures, f.fluid.pressures + f.fluid.size, 0.0f);
  std::fill(f.fluid.masses, f.fluid.masses + f.fluid.size, f.mass);

  f.cells_map[1] = new int[f.cell_count + 1];

  if (f.box_boundary) {
    f.cells_map[0] = new int[f.fluid.size];
    std::fill(f.cells_map[0], f.cells_map[0] + f.fluid.size, 0);
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
  f.boundary.colors = new glm::vec3[f.boundary.size];
  f.boundary.positions[0] = new glm::vec3[f.boundary.size];
  f.boundary.positions[1] = new glm::vec3[f.boundary.size];
  f.boundary.masses = new float[f.boundary.size];
  f.boundary.cells_lookup = new int[f.cell_count + 1];

  std::fill(f.boundary.colors, f.boundary.colors + f.boundary.size,
    glm::vec3(0.0f));
  std::copy(positions.begin(), positions.end(), f.boundary.positions[0]);
  std::fill(f.boundary.masses, f.boundary.masses + f.boundary.size, 0.0f);

  f.cells_map[0] = new int[glm::max(f.fluid.size, f.boundary.size)];
  std::fill(f.cells_map[0],
    f.cells_map[0] + glm::max(f.fluid.size, f.boundary.size), 0);
}

void SortParticles_CPU(Particles &particles, Fluid &f) {
  // Map particles to cells
  #pragma omp parallel for num_threads(f.omp_threads) \
    schedule(static, particles.size / f.omp_threads)
  for (int i = 0; i < particles.size; i++) {
    f.cells_map[0][i] = CellIndex(particles.positions[particles.swap][i] /
      f.smooth_radius, f.cell_grid);
  }

  // Initialize cells lookup
  std::fill(particles.cells_lookup,
    particles.cells_lookup + f.cell_count + 1, 0);
  #pragma omp parallel for num_threads(f.omp_threads) \
    schedule(static, particles.size / f.omp_threads)
  for (int i = 0; i < particles.size; i++) {
    assert(f.cells_map[0][i] >= 0 && f.cells_map[0][i] < f.cell_count);
    #pragma omp atomic update
    particles.cells_lookup[f.cells_map[0][i]]++;
  }

  std::exclusive_scan(particles.cells_lookup,
    particles.cells_lookup + f.cell_count + 1, particles.cells_lookup, 0);

  // Sort particles by cell index
  std::copy(particles.cells_lookup, particles.cells_lookup + f.cell_count + 1,
    f.cells_map[1]);

  int next_swap = glm::abs(particles.swap - 1);
  #pragma omp parallel for num_threads(f.omp_threads) \
    schedule(static, particles.size / f.omp_threads)
  for (int i = 0; i < particles.size; i++) {
    int index = f.cells_map[0][i];

    int offset;
    #pragma omp atomic capture
    offset = f.cells_map[1][index + 1]--;
    offset--;
    assert(offset >= 0 && offset < particles.size);

    glm::vec3 position = particles.positions[particles.swap][i];
    particles.positions[next_swap][offset] = position;
    if (particles.velocities[0] != nullptr) {
      glm::vec3 velocity = particles.velocities[particles.swap][i];
      particles.velocities[next_swap][offset] = velocity;
    }
  }

  particles.swap = next_swap;
}

void SetBoundaryMass_CPU(Fluid &f) {
  #pragma omp parallel for num_threads(f.omp_threads)
  for (int i = 0; i < f.boundary.size; i++) {
    glm::ivec3 cell(f.boundary.positions[f.boundary.swap][i] / f.smooth_radius);

    for (int x = -1; x <= 1; x++) {
      for (int y = -1; y <= 1; y++) {
        for (int z = -1; z <= 1; z++) {
          int index = CellIndex(cell + glm::ivec3(x, y, z), f.cell_grid);
          if (index < 0 || index >= f.cell_count)
            continue;

          float mass_contribution = 0.0f;
          for (int j = f.boundary.cells_lookup[index];
            j < f.boundary.cells_lookup[index + 1]; j++) {

            float distance = glm::length(
              f.boundary.positions[f.boundary.swap][i] -
              f.boundary.positions[f.boundary.swap][j]);
            float q = 2.0f * distance / f.smooth_radius;
            if (q > 2.0f || q < 1e-6f)
              continue;
            mass_contribution += MassKernel(q, f.smooth_radius);
          }
          f.boundary.masses[i] += mass_contribution;
        }
      }
    }

    f.boundary.masses[i] = 1.4f * f.density /
      glm::max(1e-6f, f.boundary.masses[i]);
  }
}

/* -------------------------------------------------------------------------- */

void ApplyGravity_CPU(Fluid &f) {
  #pragma omp parallel for num_threads(f.omp_threads) \
    schedule(static, f.omp_chunk)
  for (int i = 0; i < f.fluid.size; i++)
    f.fluid.velocities[f.fluid.swap][i].y += f.gravity * f.delta;
}

void ApplyViscosity_CPU(Fluid &f) {
  #pragma omp parallel num_threads(f.omp_threads)
  {
    #pragma omp for schedule(static, f.omp_chunk)
    for (int i = 0; i < f.fluid.size; i++) {
      glm::ivec3 cell(f.fluid.positions[f.fluid.swap][i] / f.smooth_radius);

      f.fluid.viscosities[i] = glm::vec3(0.0f);
      for (int x = -1; x <= 1; x++) {
        for (int y = -1; y <= 1; y++) {
          for (int z = -1; z <= 1; z++) {
            int index = CellIndex(cell + glm::ivec3(x, y, z), f.cell_grid);
            if (index < 0 || index >= f.cell_count)
              continue;

            glm::vec3 viscosity_contribution(0.0f);
            for (int j = f.fluid.cells_lookup[index];
              j < f.fluid.cells_lookup[index + 1]; j++) {

              float distance = glm::length(f.fluid.positions[f.fluid.swap][i] -
                f.fluid.positions[f.fluid.swap][j]);
              if (distance > f.smooth_radius)
                continue;
              viscosity_contribution += f.fluid.masses[j] *
                ((f.fluid.velocities[f.fluid.swap][j] -
                f.fluid.velocities[f.fluid.swap][i]) / f.density) *
                ViscosityKernel(distance, f.smooth_radius);
            }
            f.fluid.viscosities[i] += viscosity_contribution;
          }
        }
      }
      f.fluid.viscosities[i] *= f.viscosity;
    }

    #pragma omp barrier
    #pragma omp for schedule(static, f.omp_chunk)
    for (int i = 0; i < f.fluid.size; i++)
      f.fluid.velocities[f.fluid.swap][i] += f.fluid.viscosities[i] * f.delta;
  }
}

void ApplyDensity_CPU(Fluid &f) {
  std::fill(f.fluid.densities, f.fluid.densities + f.fluid.size, 0.0f);

  #pragma omp parallel for num_threads(f.omp_threads) \
    schedule(static, f.omp_chunk)
  for (int i = 0; i < f.fluid.size; i++) {
    glm::ivec3 cell(f.fluid.positions[f.fluid.swap][i] / f.smooth_radius);

    for (int x = -1; x <= 1; x++) {
      for (int y = -1; y <= 1; y++) {
        for (int z = -1; z <= 1; z++) {
          int index = CellIndex(cell + glm::ivec3(x, y, z), f.cell_grid);
          if (index < 0 || index >= f.cell_count)
            continue;

          // Fluid particles
          float density_contribution = 0.0f;
          for (int j = f.fluid.cells_lookup[index];
            j < f.fluid.cells_lookup[index + 1]; j++) {

            float distance = glm::length(f.fluid.positions[f.fluid.swap][i] -
              f.fluid.positions[f.fluid.swap][j]);
            float q = 2.0f * distance / f.smooth_radius;
            if (q > 2.0f || q < 1e-6f)
              continue;
            density_contribution += f.fluid.masses[j] *
              MassKernel(q, f.smooth_radius);
          }

          // Boundary particles
          if (!f.box_boundary) {
            for (int j = f.boundary.cells_lookup[index];
              j < f.boundary.cells_lookup[index + 1]; j++) {

              float distance = glm::length(f.fluid.positions[f.fluid.swap][i] -
                f.boundary.positions[f.boundary.swap][j]);
              float q = 2.0f * distance / f.smooth_radius;
              if (q > 2.0f || q < 1e-6f)
                continue;
              density_contribution += f.boundary.masses[j] *
                MassKernel(q, f.smooth_radius);
            }
          }

          f.fluid.densities[i] += density_contribution;
        }
      }
    }
  }
}

void ApplyPressure_CPU(Fluid &f) {
  #pragma omp parallel for num_threads(f.omp_threads) \
    schedule(static, f.omp_chunk)
  for (int i = 0; i < f.fluid.size; i++)
    f.fluid.pressures[i] = glm::max(0.0f, f.stiffness *
      (powf(f.fluid.densities[i] / f.density, 7.0f) - 1.0f));

  #pragma omp parallel for num_threads(f.omp_threads) \
    schedule(static, f.omp_chunk)
  for (int i = 0; i < f.fluid.size; i++) {
    glm::ivec3 cell(f.fluid.positions[f.fluid.swap][i] / f.smooth_radius);

    glm::vec3 velocity(0.0f);
    for (int x = -1; x <= 1; x++) {
      for (int y = -1; y <= 1; y++) {
        for (int z = -1; z <= 1; z++) {
          int index = CellIndex(cell + glm::ivec3(x, y, z), f.cell_grid);
          if (index < 0 || index >= f.cell_count)
            continue;

          // Fluid particles
          glm::vec3 velocity_contribution(0.0f);
          for (int j = f.fluid.cells_lookup[index];
            j < f.fluid.cells_lookup[index + 1]; j++) {

            if (i == j)
              continue;
            glm::vec3 diff = f.fluid.positions[f.fluid.swap][i] -
              f.fluid.positions[f.fluid.swap][j];
            float q = 2.0f * glm::length(diff) / f.smooth_radius;
            if (q > 2.0f)
              continue;
            velocity_contribution += -f.fluid.masses[j] *
              (f.fluid.pressures[i] /
              glm::max(1e-6f, f.fluid.densities[i] * f.fluid.densities[i]) +
              f.fluid.pressures[j] /
              glm::max(1e-6f, f.fluid.densities[j] * f.fluid.densities[j])) *
              PressureKernel(diff, q, f.smooth_radius);
          }

          // Boundary particles
          if (!f.box_boundary) {
            for (int j = f.boundary.cells_lookup[index];
              j < f.boundary.cells_lookup[index + 1]; j++) {

              glm::vec3 diff = f.fluid.positions[f.fluid.swap][i] -
                f.boundary.positions[f.boundary.swap][j];
              float q = 2.0f * glm::length(diff) / f.smooth_radius;
              if (q > 2.0f)
                continue;
              velocity_contribution += -f.boundary.masses[j] *
                (f.fluid.pressures[i] /
                glm::max(1e-6f, f.fluid.densities[i] * f.fluid.densities[i])) *
                PressureKernel(diff, q, f.smooth_radius);
            }
          }

          velocity += velocity_contribution;
        }
      }
    }

    if (glm::length(velocity) > 1000.0f)
      velocity = glm::normalize(velocity) * 1000.0f;
    f.fluid.velocities[f.fluid.swap][i] += velocity * f.delta;
  }
}

void Integrate_CPU(Fluid &f) {
  float min_bound = f.space_size - 0.999f * f.space_size;
  float max_bound = 0.999f * f.space_size;

  #pragma omp parallel for num_threads(f.omp_threads) \
    schedule(static, f.omp_chunk)
  for (int i = 0; i < f.fluid.size; i++) {
    f.fluid.positions[f.fluid.swap][i] +=
      f.fluid.velocities[f.fluid.swap][i] * f.delta;

    // Check box boundaries
    if (f.fluid.positions[f.fluid.swap][i].x < min_bound) {
      f.fluid.positions[f.fluid.swap][i].x = min_bound;
      f.fluid.velocities[f.fluid.swap][i].x =
        -f.fluid.velocities[f.fluid.swap][i].x;
    } else if (f.fluid.positions[f.fluid.swap][i].x > max_bound) {
      f.fluid.positions[f.fluid.swap][i].x = max_bound;
      f.fluid.velocities[f.fluid.swap][i].x =
        -f.fluid.velocities[f.fluid.swap][i].x;
    }
    if (f.fluid.positions[f.fluid.swap][i].y < min_bound) {
      f.fluid.positions[f.fluid.swap][i].y = min_bound;
      f.fluid.velocities[f.fluid.swap][i].y =
        -f.fluid.velocities[f.fluid.swap][i].y;
    } else if (f.fluid.positions[f.fluid.swap][i].y > max_bound) {
      f.fluid.positions[f.fluid.swap][i].y = max_bound;
      f.fluid.velocities[f.fluid.swap][i].y =
        -f.fluid.velocities[f.fluid.swap][i].y;
    }
    if (f.fluid.positions[f.fluid.swap][i].z < min_bound) {
      f.fluid.positions[f.fluid.swap][i].z = min_bound;
      f.fluid.velocities[f.fluid.swap][i].z =
        -f.fluid.velocities[f.fluid.swap][i].z;
    } else if (f.fluid.positions[f.fluid.swap][i].z > max_bound) {
      f.fluid.positions[f.fluid.swap][i].z = max_bound;
      f.fluid.velocities[f.fluid.swap][i].z =
        -f.fluid.velocities[f.fluid.swap][i].z;
    }
  }
}

void Update_CPU(Fluid &f) {
  ApplyGravity_CPU(f);
  ApplyViscosity_CPU(f);
  ApplyDensity_CPU(f);
  ApplyPressure_CPU(f);
  Integrate_CPU(f);
}

/* -------------------------------------------------------------------------- */

void Colorize_CPU(int mode, glm::vec3 base_color, Fluid &f) {
  // Base color
  if (mode == 0)
    std::fill(f.fluid.colors, f.fluid.colors + f.fluid.size, base_color);

  // Velocity
  else if (mode == 1) {
    glm::vec3 diff1(0.5f, 0.3f, 0.0f);
    glm::vec3 diff2(0.7f, -0.3f, -0.7f);
    #pragma omp parallel for num_threads(f.omp_threads) \
      schedule(static, f.omp_chunk)
    for (int i = 0; i < f.fluid.size; i++)
      f.fluid.colors[i] = ColorizeVelocity(f.fluid.velocities[f.fluid.swap][i],
        diff1, diff2, base_color);
  }

  // Pressure
  else if (mode == 2) {
    glm::vec3 diff1(0.5f, 0.3f, 0.0f);
    glm::vec3 diff2(0.7f, -0.3f, -0.7f);
    #pragma omp parallel for num_threads(f.omp_threads) \
      schedule(static, f.omp_chunk)
    for (int i = 0; i < f.fluid.size; i++)
      f.fluid.colors[i] = ColorizePressure(f.fluid.pressures[i], diff1, diff2,
        base_color);
  }

  // Density
  else if (mode == 3) {
    glm::vec3 diff1(0.5f, 0.3f, 0.0f);
    glm::vec3 diff2(0.7f, -0.3f, -0.7f);
    #pragma omp parallel for num_threads(f.omp_threads) \
      schedule(static, f.omp_chunk)
    for (int i = 0; i < f.fluid.size; i++)
      f.fluid.colors[i] = ColorizeDensity(f.fluid.densities[i], diff1, diff2,
        base_color);
  }
}

/* -------------------------------------------------------------------------- */
