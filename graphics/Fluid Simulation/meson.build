project('fluid', 'cpp', 'c',
  #'cuda', # Comment out if nvcc CUDA compiler is not available
  default_options: 'default_library=static')

# Source files
src = files(
  'source/cpu.cpp',
  'source/fluid.cpp',
  'source/main.cpp',
  'source/shader.cpp',
  'source/window.cpp')

# Dependencies
deps = [
  dependency('sdl2'),
  dependency('glew'),
  dependency('glm'),
  dependency('nlohmann_json')]

# Comment out if nvcc CUDA compiler is not available
#src += files('source/gpu.cu')
#add_project_arguments('-DUSE_CUDA', language: 'cpp')
#add_project_arguments('-DUSE_CUDA', language: 'c')
#add_project_arguments('-DUSE_CUDA', language: 'cuda')

# Executable
executable('fluid', src, dependencies: deps,
  cpp_args: ['-fopenmp'],
  link_args: ['-fopenmp', '-lGLU', '-lm'])
