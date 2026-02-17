/* CUDA C Software Rendering Pipeline ---------------------------------------
   Bachelor Thesis                                              Jakub Profota
   Summer Semester 2022/3                    OI - Computer Games and Graphics
   Supervisor Jiri Bittner               Czech Technical University in Prague
   primitive.cu ------------------------------------------------------------- */


#include "primitive.h"

#include "../core/define.h"
#include "../core/verbose.h"
#include "../platform/print.h"


static size_t device_size_indices;
static size_t device_size_triangles;
static vertex_t *device_const_vertices;
static vec3_t *device_const_normals;
static ivec2_t *device_const_indices;
static triangle_t *device_triangles;


__global__ static void primitive_assembly_kernel(
    vertex_t *, vec3_t *, ivec2_t *, size_t, triangle_t *
    #ifdef _DEBUG
    , float, int);
    #else
    );
    #endif /* _DEBUG */

__device__ static void assemble_triangle(
    size_t, vertex_t *, vec3_t *, ivec2_t *, triangle_t *);
__device__ static void backface_culling(
    size_t, triangle_t *);
__device__ static void view_frustum_clipping(
    size_t, triangle_t *);
__device__ static void clipspace_to_screenspace(
    size_t, triangle_t *);
#ifdef _DEBUG
__device__ static void degenerate_culling(
    size_t, triangle_t *, float);
#endif /* _DEBUG */


/* Init and free ------------------------------------------------------------ */


/* Initialize primitive assembly */
extern inline int primitive_init(
    vertex_t * __restrict__ vertices,   // Device vertex buffer
    vec3_t * __restrict__ normals,      // Device normal buffer
    size_t host_size_indices,           // Size of host index buffer
    ivec2_t * __restrict__ host_indices // Host index buffer
) {
    // Vertex
    device_const_vertices = vertices;

    // Normal
    device_const_normals = normals;

    // Index
    device_size_indices = host_size_indices;
    if (cudaMalloc(
        &device_const_indices, device_size_indices * sizeof(ivec2_t)
    ) != cudaSuccess) {
        #ifdef _DEBUG
        LOG_ERROR("Failed to allocate index buffer!");
        #endif /* _DEBUG */
        return EXIT_FAILURE;
    }

    // Triangle
    device_size_triangles = device_size_indices / 3;
    if (cudaMalloc(
        &device_triangles, device_size_triangles * sizeof(triangle_t)
    ) != cudaSuccess) {
        #ifdef _DEBUG
        LOG_ERROR("Failed to allocate triangle buffer!");
        #endif /* _DEBUG */
        return EXIT_FAILURE;
    }

    // Copy mesh data
    if (cudaMemcpy(
        device_const_indices,
        host_indices,
        device_size_indices * sizeof(ivec2_t),
        cudaMemcpyHostToDevice
    ) != cudaSuccess) {
        #ifdef _DEBUG
        LOG_ERROR("Failed to copy index buffer from host to device!");
        #endif /* _DEBUG */
        return EXIT_FAILURE;
    }

    #ifdef _DEBUG
    LOG_TEXT("Initialized primitive assembly");
    LOG_TRACE;
    #endif /* _DEBUG */

    return EXIT_SUCCESS;
} /* primitive_init */


/* Free primitive assembly */
extern inline int primitive_free(
    void
) {
    // Vertex
    device_const_vertices = 0;

    // Normal
    device_const_normals = 0;

    // Index
    if (cudaFree(device_const_indices) != cudaSuccess) {
        #ifdef _DEBUG
        LOG_ERROR("Failed to free index buffer!");
        #endif /* _DEBUG */
        return EXIT_FAILURE;
    }
    device_size_indices = 0;
    device_const_indices = 0;

    // Triangle
    if (cudaFree(device_triangles) != cudaSuccess) {
        #ifdef _DEBUG
        LOG_ERROR("Failed to free triangle buffer!");
        #endif /* _DEBUG */
        return EXIT_FAILURE;
    }
    device_size_triangles = 0;
    device_triangles = 0;

    #ifdef _DEBUG
    LOG_TEXT("Freed primitive assembly");
    LOG_TRACE;
    #endif /* _DEBUG */

    return EXIT_SUCCESS;
} /* primitive_free */


/* Primitive assembly ------------------------------------------------------- */


/* Assemble triangle primitives and convert coordinates */
extern inline int primitive_assembly(
    void
) {
    unsigned int warps = (unsigned int)( // Warps to process all triangles
        device_size_triangles / RENDER_WARP_SIZE + 1);
    primitive_assembly_kernel <<< warps, RENDER_WARP_SIZE >>> (
        device_const_vertices,
        device_const_normals,
        device_const_indices,
        device_size_triangles,
        device_triangles
        #ifdef _DEBUG
        , global_degenerate,
        global_culling
        #endif /* _DEBUG */
    );

    #ifdef _DEBUG
    cudaError_t error = cudaPeekAtLastError();
    if (error != cudaSuccess) {
        LOG_ERROR("%s", cudaGetErrorString(error));
        return EXIT_FAILURE;
    }
    #endif /* _DEBUG */

    cudaDeviceSynchronize();

    return EXIT_SUCCESS;
} /* primitive_assembly */


/* Kernel ------------------------------------------------------------------- */


/* Assemble triangle primitives, cull, clip and convert coordinates */
__global__ static void primitive_assembly_kernel(
    vertex_t * __restrict__ ver,  // Vertex buffer
    vec3_t * __restrict__ nor,    // Normal buffer
    ivec2_t * __restrict__ ind,   // Index buffer
    size_t s_tri,                 // Size of triangle buffer
    triangle_t * __restrict__ tri // Triangle buffer
    #ifdef _DEBUG
    , float degenerate_factor,    // Maximum area of degenerate triangle
    int global_culling            // Culling
    #endif /* _DEBUG */
) {
    size_t id = (size_t)(blockIdx.x * blockDim.x + threadIdx.x);
    if (id < s_tri) {
        assemble_triangle(id, ver, nor, ind, tri);
        #ifdef _DEBUG
        if (global_culling & CULLING_BACKFACE) {
        #endif /* _DEBUG */
            backface_culling(id, tri);
        #ifdef _DEBUG
        }
        if (global_culling & CULLING_FRUSTUM) {
        #endif /* _DEBUG */
            view_frustum_clipping(id, tri);
        #ifdef _DEBUG
        }
        #endif /* _DEBUG */
        clipspace_to_screenspace(id, tri);
        #ifdef _DEBUG
        degenerate_culling(id, tri, degenerate_factor);
        #endif /* _DEBUG */
    }
} /* primitive_assembly_triangle_kernel */


/* Assemble triangle primitives */
__device__ static void assemble_triangle(
    size_t id,      // Thread id
    vertex_t * __restrict__ ver,  // Vertex buffer
    vec3_t * __restrict__ nor,    // Normal buffer
    ivec2_t * __restrict__ ind,   // Index buffer
    triangle_t * __restrict__ tri // Triangle buffer
) {
    // Vertex
    vec3_t v0, l0, h0;
    v0[0] = ver[ind[id * 3 + 0][0] - 1].position[0];
    v0[1] = ver[ind[id * 3 + 0][0] - 1].position[1];
    v0[2] = ver[ind[id * 3 + 0][0] - 1].position[2];
    l0[0] = ver[ind[id * 3 + 0][0] - 1].light[0];
    l0[1] = ver[ind[id * 3 + 0][0] - 1].light[1];
    l0[2] = ver[ind[id * 3 + 0][0] - 1].light[2];
    h0[0] = ver[ind[id * 3 + 0][0] - 1].half[0];
    h0[1] = ver[ind[id * 3 + 0][0] - 1].half[1];
    h0[2] = ver[ind[id * 3 + 0][0] - 1].half[2];
    vec3_t v1, l1, h1;
    v1[0] = ver[ind[id * 3 + 1][0] - 1].position[0];
    v1[1] = ver[ind[id * 3 + 1][0] - 1].position[1];
    v1[2] = ver[ind[id * 3 + 1][0] - 1].position[2];
    l1[0] = ver[ind[id * 3 + 1][0] - 1].light[0];
    l1[1] = ver[ind[id * 3 + 1][0] - 1].light[1];
    l1[2] = ver[ind[id * 3 + 1][0] - 1].light[2];
    h1[0] = ver[ind[id * 3 + 1][0] - 1].half[0];
    h1[1] = ver[ind[id * 3 + 1][0] - 1].half[1];
    h1[2] = ver[ind[id * 3 + 1][0] - 1].half[2];
    vec3_t v2, l2, h2;
    v2[0] = ver[ind[id * 3 + 2][0] - 1].position[0];
    v2[1] = ver[ind[id * 3 + 2][0] - 1].position[1];
    v2[2] = ver[ind[id * 3 + 2][0] - 1].position[2];
    l2[0] = ver[ind[id * 3 + 2][0] - 1].light[0];
    l2[1] = ver[ind[id * 3 + 2][0] - 1].light[1];
    l2[2] = ver[ind[id * 3 + 2][0] - 1].light[2];
    h2[0] = ver[ind[id * 3 + 2][0] - 1].half[0];
    h2[1] = ver[ind[id * 3 + 2][0] - 1].half[1];
    h2[2] = ver[ind[id * 3 + 2][0] - 1].half[2];

    // Normal
    vec3_t n0;
    n0[0] = nor[ind[id * 3 + 0][1] - 1][0];
    n0[1] = nor[ind[id * 3 + 0][1] - 1][1];
    n0[2] = nor[ind[id * 3 + 0][1] - 1][2];
    vec3_t n1;
    n1[0] = nor[ind[id * 3 + 1][1] - 1][0];
    n1[1] = nor[ind[id * 3 + 1][1] - 1][1];
    n1[2] = nor[ind[id * 3 + 1][1] - 1][2];
    vec3_t n2;
    n2[0] = nor[ind[id * 3 + 2][1] - 1][0];
    n2[1] = nor[ind[id * 3 + 2][1] - 1][1];
    n2[2] = nor[ind[id * 3 + 2][1] - 1][2];

    // Triangle
    tri[id].vertices[0].position[0] = v0[0];
    tri[id].vertices[0].position[1] = v0[1];
    tri[id].vertices[0].position[2] = v0[2];
    tri[id].vertices[0].light[0] = l0[0];
    tri[id].vertices[0].light[1] = l0[1];
    tri[id].vertices[0].light[2] = l0[2];
    tri[id].vertices[0].half[0] = h0[0];
    tri[id].vertices[0].half[1] = h0[1];
    tri[id].vertices[0].half[2] = h0[2];
    tri[id].normals[0][0] = n0[0];
    tri[id].normals[0][1] = n0[1];
    tri[id].normals[0][2] = n0[2];

    tri[id].vertices[1].position[0] = v1[0];
    tri[id].vertices[1].position[1] = v1[1];
    tri[id].vertices[1].position[2] = v1[2];
    tri[id].vertices[1].light[0] = l1[0];
    tri[id].vertices[1].light[1] = l1[1];
    tri[id].vertices[1].light[2] = l1[2];
    tri[id].vertices[1].half[0] = h1[0];
    tri[id].vertices[1].half[1] = h1[1];
    tri[id].vertices[1].half[2] = h1[2];
    tri[id].normals[1][0] = n1[0];
    tri[id].normals[1][1] = n1[1];
    tri[id].normals[1][2] = n1[2];

    tri[id].vertices[2].position[0] = v2[0];
    tri[id].vertices[2].position[1] = v2[1];
    tri[id].vertices[2].position[2] = v2[2];
    tri[id].vertices[2].light[0] = l2[0];
    tri[id].vertices[2].light[1] = l2[1];
    tri[id].vertices[2].light[2] = l2[2];
    tri[id].vertices[2].half[0] = h2[0];
    tri[id].vertices[2].half[1] = h2[1];
    tri[id].vertices[2].half[2] = h2[2];
    tri[id].normals[2][0] = n2[0];
    tri[id].normals[2][1] = n2[1];
    tri[id].normals[2][2] = n2[2];
} /* assemble_triangle */


/* Backface culling */
__device__ static void backface_culling(
    size_t id,                    // Thread id
    triangle_t * __restrict__ tri // Triangle buffer
) {
    vec2_t v1;
    v1[0] = tri[id].vertices[1].position[0] - tri[id].vertices[0].position[0];
    v1[1] = tri[id].vertices[1].position[1] - tri[id].vertices[0].position[1];

    vec2_t v2;
    v2[0] = tri[id].vertices[2].position[0] - tri[id].vertices[0].position[0];
    v2[1] = tri[id].vertices[2].position[1] - tri[id].vertices[0].position[1];

    if (v1[0] * v2[1] - v1[1] * v2[0] < 0.0f) {
        tri[id].discard[0] = 1;
    } else {
        tri[id].discard[0] = 0;
    }
} /* backface_culling */


/* View frustum clipping */
__device__ static void view_frustum_clipping(
    size_t id,                    // Thread id
    triangle_t * __restrict__ tri // Triangle buffer
) {
    vec3_t v_min, v_max;
    triangle_aabb_3d(tri[id], v_min, v_max);

    if (v_min[0] > 1.0f || v_min[1] > 1.0f ||
        v_max[0] < -1.0f || v_max[1] < -1.0f ||
        v_min[2] < 0.0f || v_min[2] > 1.0f ||
        v_max[2] < 0.0f || v_max[2] > 1.0f) {
        tri[id].discard[1] = 1;
    } else {
        tri[id].discard[1] = 0;
    }
} /* view_frustum_clipping */


/* Convert clip space coordinates to screen space coordinates */
__device__ static void clipspace_to_screenspace(
    size_t id,                    // Thread id
    triangle_t * __restrict__ tri // Triangle buffer
) {
    tri[id].vertices[0].position[0] =
        (tri[id].vertices[0].position[0] + 1.0f) * 0.5f *
        (float)WINDOW_WIDTH;
    tri[id].vertices[0].position[1] =
        (tri[id].vertices[0].position[1] + 1.0f) * 0.5f *
        (float)WINDOW_HEIGHT;
    tri[id].vertices[0].position[2] =
        (tri[id].vertices[0].position[2] + 1.0f) * 0.5f;

    tri[id].vertices[1].position[0] =
        (tri[id].vertices[1].position[0] + 1.0f) * 0.5f *
        (float)WINDOW_WIDTH;
    tri[id].vertices[1].position[1] =
        (tri[id].vertices[1].position[1] + 1.0f) * 0.5f *
        (float)WINDOW_HEIGHT;
    tri[id].vertices[1].position[2] =
        (tri[id].vertices[1].position[2] + 1.0f) * 0.5f;

    tri[id].vertices[2].position[0] =
        (tri[id].vertices[2].position[0] + 1.0f) * 0.5f *
        (float)WINDOW_WIDTH;
    tri[id].vertices[2].position[1] =
        (tri[id].vertices[2].position[1] + 1.0f) * 0.5f *
        (float)WINDOW_HEIGHT;
    tri[id].vertices[2].position[2] =
        (tri[id].vertices[2].position[2] + 1.0f) * 0.5f;
} /* clipspace_to_screenspace */


#ifdef _DEBUG


/* Degenerate culling */
__device__ static void degenerate_culling(
    size_t id,                     // Thread id
    triangle_t * __restrict__ tri, // Triangle buffer
    float degenerate_factor        // Maximum area of degenerate triangle
) {
    vec2_t v0;
    v0[0] = tri[id].vertices[0].position[0];
    v0[1] = tri[id].vertices[0].position[1];
    vec2_t v1;
    v1[0] = tri[id].vertices[1].position[0];
    v1[1] = tri[id].vertices[1].position[1];
    vec2_t v2;
    v2[0] = tri[id].vertices[2].position[0];
    v2[1] = tri[id].vertices[2].position[1];

    float area = (v0[0] * v1[1] + v1[0] * v2[1] + v0[1] * v2[0] -
                  v2[0] * v1[1] - v0[1] * v1[0] - v0[0] * v2[1]) * 0.5f;
    if (-degenerate_factor < area && area < degenerate_factor) {
        tri[id].discard[2] = 1;
    } else {
        tri[id].discard[2] = 0;
    }
} /* degenerate_culling */


#endif /* _DEBUG */


/* Get ---------------------------------------------------------------------- */


/* Get size of triangle buffer */
size_t primitive_get_size_triangles(
    void
) {
    return device_size_triangles;
} /* primitive_get_size_triangles */


/* Get triangle buffer */
extern inline triangle_t *primitive_get_triangles(
    void
) {
    return device_triangles;
} /* primitive_get_triangles */


/* Verbose ------------------------------------------------------------------ */


#ifdef _DEBUG


/* Verbose output */
extern void primitive_verbose(
    long long time // Elapsed time
) {
    LOG_TEXT("Primitive assembly");
    LOG_TRACE;
    print(PRINT_LIME_BG, " %lld ", time);
    print(PRINT_GREEN_BG, " us \n");

    ivec2_t *reference_indices = (ivec2_t *)malloc(
        device_size_indices * sizeof(ivec2_t));
    triangle_t *reference_triangles = (triangle_t *)malloc(
        device_size_triangles * sizeof(triangle_t));

    cudaMemcpy(
        reference_indices,
        device_const_indices,
        device_size_indices * sizeof(ivec2_t),
        cudaMemcpyDeviceToHost
    );
    cudaMemcpy(
        reference_triangles,
        device_triangles,
        device_size_triangles * sizeof(triangle_t),
        cudaMemcpyDeviceToHost
    );

    size_t backface_culled = 0;
    size_t view_frustum_clipped = 0;
    size_t degenerate_culled = 0;
    for (size_t i = 0; i < device_size_triangles; i++) {
        if (reference_triangles[i].discard[0]) {
            backface_culled++;
        }
        if (reference_triangles[i].discard[1]) {
            view_frustum_clipped++;
        }
        if (reference_triangles[i].discard[2]) {
            degenerate_culled++;
        }
    }

    print(PRINT_CYAN_BG, " Buffer \n");
    print(PRINT_AQUA_BG, " %llu triangles \n", device_size_triangles);
    print(PRINT_BLUE_BG, " Culling and clipping \n");
    print(PRINT_WHITE_FG, " %llu backface, %llu view frustum, "
        "%llu degenerate \n",
        backface_culled, view_frustum_clipped, degenerate_culled);
    for (size_t i = 0; i < device_size_triangles; i++) {
        if (i == 0 || i == device_size_triangles - 1) {
            print(PRINT_BLUE_BG, " %llu \n", i + 1);
            print(PRINT_GRAY_FG, "          vertices\n");
            verbose_vec3(reference_triangles[i].vertices[0].position);
            verbose_vec3(reference_triangles[i].vertices[1].position);
            verbose_vec3(reference_triangles[i].vertices[2].position);
            print(PRINT_GRAY_FG, "           normals\n");
            verbose_vec3(reference_triangles[i].normals[0]);
            verbose_vec3(reference_triangles[i].normals[1]);
            verbose_vec3(reference_triangles[i].normals[2]);
        } else if (i == 1) {
            print(PRINT_GRAY_FG, " . . .\n");
        }
    }

    free(reference_indices);
    free(reference_triangles);
} /* primitive_verbose */


#endif /* _DEBUG */


/* -------------------------------------------------------------------------- */


/* primitive.cu */