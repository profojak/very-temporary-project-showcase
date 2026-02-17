/* CUDA C Software Rendering Pipeline ---------------------------------------
   Bachelor Thesis                                              Jakub Profota
   Summer Semester 2022/3                    OI - Computer Games and Graphics
   Supervisor Jiri Bittner               Czech Technical University in Prague
   define.h ----------------------------------------------------------------- */


#ifndef DEFINE_H
#define DEFINE_H


#define MATH_PI 3.14159265358979323846264338327950288f // Number pi

#define MESH_BATCH_SIZE 1024 // Reallocation factor when loading mesh

#define SCENE_VALUE_RESET 3.402823466e+38f // Passed to reset to defaults

#define PRINT_MAX_LEN 80 // Maximum width of console output line

#define WINDOW_WIDTH 1280  // Window width
#define WINDOW_HEIGHT 1024 // Window height
#define WINDOW_ASPECT_RATIO ((float)WINDOW_WIDTH / (float)WINDOW_HEIGHT)
#define WINDOW_FPS 60     // Target frames per second

#define RENDER_WARP_SIZE 32           // Threads in CUDA warp
#define RENDER_MAX_THREADS_BLOCK 1024 // Maximum threads per block
#define RENDER_BIN_SIZE 8             // Tiles per bin width
#define RENDER_TILE_SIZE 16           // Fragments per tile width
#define RENDER_BIN_WIDTH (WINDOW_WIDTH / (RENDER_BIN_SIZE * RENDER_TILE_SIZE))
#define RENDER_BIN_HEIGHT (WINDOW_HEIGHT / (RENDER_BIN_SIZE * RENDER_TILE_SIZE))

#define BENCHMARK_IDLE_FRAMES WINDOW_FPS * 5 // Length of idle benchmark
#define BENCHMARK_360_FRAMES 720             // Length of 360 degrees benchmark
#define BENCHMARK_ZOOM_FRAMES 720            // Length of zoom benchmark


#ifndef _DEBUG
#define RENDER_BIN_BUFFER 524288  // Size of bin rasterization buffer
#define RENDER_TILE_BUFFER 131072 // Size of tile rasterization buffer
#endif /* _DEBUG */


// Command line arguments and runtime globals
extern int global_align;     // Align window and console
extern int global_layout;    // Swap z and y keys
extern char *global_mesh;    // Currently loaded mesh
extern int global_verbose;   // Verbose output
#define VERBOSE_ENABLED 1    // Verbose output enabled
#define VERBOSE_QUEUED 2     // Verbose output queued for next render pass
#define VERBOSE_ACTIVE 3     // Verbose output for current render pass in process
extern int global_benchmark; // Benchmark
#define BENCHMARK_IDLE 1     // Idle benchmark
#define BENCHMARK_360 2      // 360 degrees benchmark
#define BENCHMARK_ZOOM 3     // Zoom benchmark
#ifdef _DEBUG
extern int global_bin;          // Size of bin rasterization buffer
extern int global_tile;         // Size of tile rasterization buffer
extern float global_degenerate; // Maximum area of degenerate triangle
extern int global_primitive;    // Primitive draw type
#define PRIMITIVE_TRIANGLE 1    // Draw triangles
#define PRIMITIVE_LINE 2        // Draw lines
#define PRIMITIVE_POINT 3       // Draw points
extern int global_shading;      // Shading
#define SHADING_FLAT 1          // Flat shading
#define SHADING_SMOOTH 2        // Smooth shading
extern int global_color;        // Coloring of triangles
#define COLOR_AMBIENT 1         // Ambient component only
#define COLOR_DIFFUSE 2         // Diffuse component only
#define COLOR_SPECULAR 3        // Specular component only
#define COLOR_BLINN_PHONG 4     // Blinn-Phong illumination
#define COLOR_CAMERA 5          // Camera reflector illumination
#define COLOR_NORMAL 6          // Normal vectors
#define COLOR_DEPTH 7           // Depth buffer
extern int global_culling;      // Culling
#define CULLING_BACKFACE 1      // Backface culling
#define CULLING_FRUSTUM 2       // View frustum clipping
#endif /* _DEBUG */


/* -------------------------------------------------------------------------- */


#endif /* DEFINE_H */


/* define.h */