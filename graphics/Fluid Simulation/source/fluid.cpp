/// \file fluid.cpp
/// \brief Fluid simulation implementation.

#include "fluid.hpp"

// CPU functions

/// \brief Destroys the fluid simulation on the CPU.
/// \param f Fluid simulation.
extern void Destroy_CPU(Fluid &f);
/// \brief Places fluid and boundary particles, if configured,
/// in the simulation space on the CPU.
/// \param f Fluid simulation.
extern void PlaceParticles_CPU(Fluid &f);
/// \brief Sorts particles into cells on the CPU.
/// \param particles Particles to sort.
/// \param f Fluid simulation.
extern void SortParticles_CPU(Particles &particles, Fluid &f);
/// \brief Sets boundary particles mass on the CPU.
/// \param f Fluid simulation.
extern void SetBoundaryMass_CPU(Fluid &f);
/// \brief Advances fluid simulation by one simulation time step on the CPU.
/// \param f Fluid simulation.
extern void Update_CPU(Fluid &f);
/// \brief Colorizes fluid particles based on their properties on the CPU.
/// \param mode Colorization mode.
/// \param base_color Base color.
/// \param f Fluid simulation.
extern void Colorize_CPU(int mode, glm::vec3 base_color, Fluid &f);

#ifdef USE_CUDA
// GPU functions

/// \brief Destroys the fluid simulation on the GPU.
/// \param f Fluid simulation.
extern void Destroy_GPU(Fluid &f);
/// \brief Sets up the fluid simulation on the GPU.
/// \param f Fluid simulation.
extern void Setup_GPU(Fluid &f);
/// \brief Places fluid and boundary particles, if configured,
/// in the simulation space on the GPU.
/// \param f Fluid simulation.
extern void PlaceParticles_GPU(Fluid &f);
/// \brief Sorts particles into cells on the GPU.
/// \param particles Particles to sort.
/// \param f Fluid simulation.
extern void SortParticles_GPU(Particles &particles, Fluid &f);
/// \brief Sets boundary particles mass on the GPU.
/// \param f Fluid simulation.
extern void SetBoundaryMass_GPU(Fluid &f);
/// \brief Advances fluid simulation by one simulation time step on the GPU.
/// \param f Fluid simulation.
extern void Update_GPU(Fluid &f);
/// \brief Colorizes fluid particles based on their properties on the GPU.
/// \param mode Colorization mode.
/// \param base_color Base color.
/// \param f Fluid simulation.
extern void Colorize_GPU(int mode, glm::vec3 base_color, Fluid &f);
#endif

/* -------------------------------------------------------------------------- */

void Fluid::Create(nlohmann::json &config) {
  delta = config["simulation"]["delta"].get<float>();

  gpu = config["compute"]["gpu"].get<bool>();
  gpu_threads = config["compute"]["gpu_threads"].get<int>();
  omp_threads = config["compute"]["omp_threads"].get<int>();

  space_size = config["simulation"]["space_size"].get<float>();
  smooth_radius = config["simulation"]["smooth_radius"].get<float>();
  for (int i = 0; i < 3; i++)
    fluid_grid[i] = config["simulation"]["fluid_grid"][i].get<int>();
  cell_grid = glm::ivec3(ceil(space_size / smooth_radius));
  cell_count = cell_grid.x * cell_grid.y * cell_grid.z;
  box_boundary = config["simulation"]["box_boundary"].get<bool>();

  gravity = config["fluid"]["gravity"].get<float>();
  mass = config["fluid"]["mass"].get<float>();
  density = config["fluid"]["density"].get<float>();
  viscosity = config["fluid"]["viscosity"].get<float>();
  stiffness = config["fluid"]["stiffness"].get<float>();

  fluid.size = 0;
  fluid.colors = nullptr;
  fluid.positions[0] = nullptr;
  fluid.positions[1] = nullptr;
  fluid.velocities[0] = nullptr;
  fluid.velocities[1] = nullptr;
  fluid.viscosities = nullptr;
  fluid.densities = nullptr;
  fluid.pressures = nullptr;
  fluid.masses = nullptr;
  fluid.cells_lookup = nullptr;

  boundary.size = 0;
  boundary.colors = nullptr;
  boundary.positions[0] = nullptr;
  boundary.positions[1] = nullptr;
  boundary.velocities[0] = nullptr;
  boundary.velocities[1] = nullptr;
  boundary.viscosities = nullptr;
  boundary.densities = nullptr;
  boundary.pressures = nullptr;
  boundary.masses = nullptr;
  boundary.cells_lookup = nullptr;

  cells_map[0] = nullptr;
  cells_map[1] = nullptr;

  is_init = false;
#ifdef USE_CUDA
  gpu_init = false;
#endif
}

void Fluid::Destroy() {
#ifdef USE_CUDA
  if (gpu) {
    Destroy_GPU(*this);
    return;
  }
#endif

  Destroy_CPU(*this);
}

/* -------------------------------------------------------------------------- */

void Fluid::Reset() {
  total_time = 0.0f;
  steps = 0;
  omp_chunk = 0;

  fluid.swap = 0;
  boundary.swap = 0;

  if (is_init)
    Destroy();

  PlaceParticles();
  Colorize(0);
  if (!box_boundary) {
    SortParticles(boundary);
    SetBoundaryMass();
  }

#ifdef USE_CUDA
  if (gpu && !gpu_init)
    Setup_GPU(*this);
#endif
}

void Fluid::PlaceParticles() {
  is_init = true;

#ifdef USE_CUDA
  if (gpu) {
    PlaceParticles_GPU(*this);
    return;
  }
#endif

  PlaceParticles_CPU(*this);
}

void Fluid::SortParticles(Particles &particles) {
#ifdef USE_CUDA
  if (gpu) {
    SortParticles_GPU(particles, *this);
    return;
  }
#endif

  SortParticles_CPU(particles, *this);
}

void Fluid::SetBoundaryMass() {
#ifdef USE_CUDA
  if (gpu) {
    SetBoundaryMass_GPU(*this);
    return;
  }
#endif

  SetBoundaryMass_CPU(*this);
}

/* -------------------------------------------------------------------------- */

void Fluid::Update() {
  steps++;
  total_time += delta;

  SortParticles(fluid);

#ifdef USE_CUDA
  if (gpu) {
    Update_GPU(*this);
    return;
  }
#endif

  Update_CPU(*this);
}

void Fluid::Colorize(int mode) {
  glm::vec3 base_color(0.3f, 0.5f, 0.95f);

#ifdef USE_CUDA
  if (gpu) {
    Colorize_GPU(mode, base_color, *this);
    return;
  }
#endif

  Colorize_CPU(mode, base_color, *this);
}

/* -------------------------------------------------------------------------- */
