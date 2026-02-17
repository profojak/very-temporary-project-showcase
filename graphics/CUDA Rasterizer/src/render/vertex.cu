/* CUDA C Software Rendering Pipeline ---------------------------------------
   Bachelor Thesis                                              Jakub Profota
   Summer Semester 2022/3                    OI - Computer Games and Graphics
   Supervisor Jiri Bittner               Czech Technical University in Prague
   vertex.cu ---------------------------------------------------------------- */


#include "vertex.h"

#include "../core/define.h"
#include "../core/verbose.h"
#include "../platform/print.h"


static size_t device_size_vertices;
static size_t device_size_normals;
static vec3_t *device_const_vertices;
static vec3_t *device_const_normals;
static vertex_t *device_vertices;
static vec3_t *device_normals;
static vec3_t host_light_position;
static mat4_t host_view;
static mat4_t host_view_model;
static mat4_t host_view_model_inverse_transpose;
static mat4_t host_perspective_view_model;
static vec3_t *device_light_position;
static mat4_t *device_view;
static mat4_t *device_view_model;
static mat4_t *device_view_model_inverse_transpose;
static mat4_t *device_perspective_view_model;


__global__ static void vertex_shader_vertices_kernel(
    mat4_t *, mat4_t *, mat4_t *, size_t, vec3_t *, vertex_t *, vec3_t *);
__global__ static void vertex_shader_normals_kernel(
    mat4_t *, size_t, vec3_t *, vec3_t *);


/* Init and free ------------------------------------------------------------ */


/* Initialize vertex shader */
extern inline int vertex_init(
    size_t host_size_vertices,           // Size of host vertex buffer
    vec3_t * __restrict__ host_vertices, // Host vertex buffer
    size_t host_size_normals,            // Size of host normal buffer
    vec3_t * __restrict__ host_normals   // Host normal buffer
) {
    // Vertex
    device_size_vertices = host_size_vertices;
    if (cudaMalloc(
        &device_const_vertices, device_size_vertices * sizeof(vec3_t)
    ) != cudaSuccess) {
        #ifdef _DEBUG
        LOG_ERROR("Failed to allocate vertex buffer!");
        #endif /* _DEBUG */
        return EXIT_FAILURE;
    }

    if (cudaMalloc(
        &device_vertices, device_size_vertices * sizeof(vertex_t)
    ) != cudaSuccess) {
        #ifdef _DEBUG
        LOG_ERROR("Failed to allocate vertex buffer!");
        #endif /* _DEBUG */
        return EXIT_FAILURE;
    }

    // Normal
    device_size_normals = host_size_normals;
    if (cudaMalloc(
        &device_const_normals, device_size_normals * sizeof(vec3_t)
    ) != cudaSuccess) {
        #ifdef _DEBUG
        LOG_ERROR("Failed to allocate normal buffer!");
        #endif /* _DEBUG */
        return EXIT_FAILURE;
    }

    if (cudaMalloc(
        &device_normals, device_size_normals * sizeof(vec3_t)
    ) != cudaSuccess) {
        #ifdef _DEBUG
        LOG_ERROR("Failed to allocate normal buffer!");
        #endif /* _DEBUG */
        return EXIT_FAILURE;
    }

    // Light
    if (cudaMalloc(
        &device_light_position, sizeof(vec3_t)
    ) != cudaSuccess) {
        #ifdef _DEBUG
        LOG_ERROR("Failed to allocate light position!");
        #endif /* _DEBUG */
        return EXIT_FAILURE;
    }

    // Matrix
    if (cudaMalloc(
        &device_view, sizeof(mat4_t)
    ) != cudaSuccess) {
        #ifdef _DEBUG
        LOG_ERROR("Failed to allocate transformation matrix!");
        #endif /* _DEBUG */
        return EXIT_FAILURE;
    }

    if (cudaMalloc(
        &device_view_model, sizeof(mat4_t)
    ) != cudaSuccess) {
        #ifdef _DEBUG
        LOG_ERROR("Failed to allocate transformation matrix!");
        #endif /* _DEBUG */
        return EXIT_FAILURE;
    }

    if (cudaMalloc(
        &device_view_model_inverse_transpose, sizeof(mat4_t)
    ) != cudaSuccess) {
        #ifdef _DEBUG
        LOG_ERROR("Failed to allocate transformation matrix!");
        #endif /* _DEBUG */
        return EXIT_FAILURE;
    }

    if (cudaMalloc(
        &device_perspective_view_model, sizeof(mat4_t)
    ) != cudaSuccess) {
        #ifdef _DEBUG
        LOG_ERROR("Failed to allocate transformation matrix!");
        #endif /* _DEBUG */
        return EXIT_FAILURE;
    }

    // Copy mesh data
    if (cudaMemcpy(
        device_const_vertices,
        host_vertices,
        device_size_vertices * sizeof(vec3_t),
        cudaMemcpyHostToDevice
    ) != cudaSuccess) {
        #ifdef _DEBUG
        LOG_ERROR("Failed to copy vertex buffer from host to device!");
        #endif /* _DEBUG */
        return EXIT_FAILURE;
    }

    if (cudaMemcpy(
        device_const_normals,
        host_normals,
        device_size_normals * sizeof(vec3_t),
        cudaMemcpyHostToDevice
    ) != cudaSuccess) {
        #ifdef _DEBUG
        LOG_ERROR("Failed to copy normal buffer from host to device!");
        #endif /* _DEBUG */
        return EXIT_FAILURE;
    }

    #ifdef _DEBUG
    LOG_TEXT("Initialized vertex shader");
    LOG_TRACE;
    #endif /* _DEBUG */

    return EXIT_SUCCESS;
} /* vertex_init */


/* Free vertex shader */
extern inline int vertex_free(
    void
) {
    // Vertex
    if (cudaFree(device_const_vertices) != cudaSuccess) {
        #ifdef _DEBUG
        LOG_ERROR("Failed to free vertex buffer!");
        #endif /* _DEBUG */
        return EXIT_FAILURE;
    }
    device_size_vertices = 0;
    device_const_vertices = 0;

    if (cudaFree(device_vertices) != cudaSuccess) {
        #ifdef _DEBUG
        LOG_ERROR("Failed to free vertex buffer!");
        #endif /* _DEBUG */
        return EXIT_FAILURE;
    }
    device_vertices = 0;

    // Normal
    if (cudaFree(device_const_normals) != cudaSuccess) {
        #ifdef _DEBUG
        LOG_ERROR("Failed to free normal buffer!");
        #endif /* _DEBUG */
        return EXIT_FAILURE;
    }
    device_size_normals = 0;
    device_const_normals = 0;

    if (cudaFree(device_normals) != cudaSuccess) {
        #ifdef _DEBUG
        LOG_ERROR("Failed to free normal buffer!");
        #endif /* _DEBUG */
        return EXIT_FAILURE;
    }
    device_normals = 0;

    // Light
    if (cudaFree(device_light_position) != cudaSuccess) {
        #ifdef _DEBUG
        LOG_ERROR("Failed to free light position!");
        #endif /* _DEBUG */
        return EXIT_FAILURE;
    }
    device_light_position = 0;

    // Matrix
    if (cudaFree(device_view) != cudaSuccess) {
        #ifdef _DEBUG
        LOG_ERROR("Failed to free transformation matrix!");
        #endif /* _DEBUG */
        return EXIT_FAILURE;
    }
    device_view = 0;

    if (cudaFree(device_view_model) != cudaSuccess) {
        #ifdef _DEBUG
        LOG_ERROR("Failed to free transformation matrix!");
        #endif /* _DEBUG */
        return EXIT_FAILURE;
    }
    device_view_model = 0;

    if (cudaFree(device_view_model_inverse_transpose) != cudaSuccess) {
        #ifdef _DEBUG
        LOG_ERROR("Failed to free transformation matrix!");
        #endif /* _DEBUG */
        return EXIT_FAILURE;
    }
    device_view_model_inverse_transpose = 0;

    if (cudaFree(device_perspective_view_model) != cudaSuccess) {
        #ifdef _DEBUG
        LOG_ERROR("Failed to free transformation matrix!");
        #endif /* _DEBUG */
        return EXIT_FAILURE;
    }
    device_perspective_view_model = 0;

    #ifdef _DEBUG
    LOG_TEXT("Freed vertex shader");
    LOG_TRACE;
    #endif /* _DEBUG */

    return EXIT_SUCCESS;
} /* vertex_free */


/* Vertex shader ------------------------------------------------------------ */


/* Transform vertices and normals */
extern inline int vertex_shader(
    void
) {
    if (cudaMemcpy(
        device_light_position,
        host_light_position,
        sizeof(vec3_t),
        cudaMemcpyHostToDevice
    ) != cudaSuccess) {
        #ifdef _DEBUG
        LOG_ERROR("Failed to copy light position from host to device!");
        #endif /* _DEBUG */
        return EXIT_FAILURE;
    }

    if (cudaMemcpy(
        device_view,
        host_view,
        sizeof(mat4_t),
        cudaMemcpyHostToDevice
    ) != cudaSuccess) {
        #ifdef _DEBUG
        LOG_ERROR("Failed to copy transformation matrix from host to device!");
        #endif /* _DEBUG */
        return EXIT_FAILURE;
    }

    if (cudaMemcpy(
        device_view_model,
        host_view_model,
        sizeof(mat4_t),
        cudaMemcpyHostToDevice
    ) != cudaSuccess) {
        #ifdef _DEBUG
        LOG_ERROR("Failed to copy transformation matrix from host to device!");
        #endif /* _DEBUG */
        return EXIT_FAILURE;
    }

    if (cudaMemcpy(
        device_view_model_inverse_transpose,
        host_view_model_inverse_transpose,
        sizeof(mat4_t),
        cudaMemcpyHostToDevice
    ) != cudaSuccess) {
        #ifdef _DEBUG
        LOG_ERROR("Failed to copy transformation matrix from host to device!");
        #endif /* _DEBUG */
        return EXIT_FAILURE;
    }

    if (cudaMemcpy(
        device_perspective_view_model,
        host_perspective_view_model,
        sizeof(mat4_t),
        cudaMemcpyHostToDevice
    ) != cudaSuccess) {
        #ifdef _DEBUG
        LOG_ERROR("Failed to copy transformation matrix from host to device!");
        #endif /* _DEBUG */
        return EXIT_FAILURE;
    }

    // Vertex
    unsigned int warps = (unsigned int)( // Warps to process all vertices
        device_size_vertices / RENDER_WARP_SIZE + 1);
    vertex_shader_vertices_kernel <<< warps, RENDER_WARP_SIZE >>> (
        device_view,
        device_view_model,
        device_perspective_view_model,
        device_size_vertices,
        device_const_vertices,
        device_vertices,
        device_light_position
    );

    #ifdef _DEBUG
    cudaError_t error = cudaPeekAtLastError();
    if (error != cudaSuccess) {
        LOG_ERROR("%s", cudaGetErrorString(error));
        return EXIT_FAILURE;
    }
    #endif /* _DEBUG */

    // Normal
    warps = (unsigned int)( // Warps to process all normals
        device_size_normals / RENDER_WARP_SIZE + 1);
    vertex_shader_normals_kernel <<< warps, RENDER_WARP_SIZE >>> (
        device_view_model_inverse_transpose,
        device_size_normals,
        device_const_normals,
        device_normals
    );

    #ifdef _DEBUG
    error = cudaPeekAtLastError();
    if (error != cudaSuccess) {
        LOG_ERROR("%s", cudaGetErrorString(error));
        return EXIT_FAILURE;
    }
    #endif /* _DEBUG */

    cudaDeviceSynchronize();

    return EXIT_SUCCESS;
} /* vertex_shader */


/* Kernel ------------------------------------------------------------------- */


/* Transform vertices */
__global__ static void vertex_shader_vertices_kernel(
    mat4_t * __restrict__ v,         // View transformation matrix
    mat4_t * __restrict__ vm,        // View model transformation matrix
    mat4_t * __restrict__ pvm,       // Perspective view model transformation
    size_t s_ver,                    // Size of vertex buffer
    vec3_t * __restrict__ const_ver, // Vertex buffer
    vertex_t * __restrict__ ver,     // Target vertex buffer
    vec3_t * __restrict__ light_pos  // Light position
) {
    size_t id = (size_t)(blockIdx.x * blockDim.x + threadIdx.x);
    if (id < s_ver) {
        vec4_t c_ver;
        c_ver[0] = const_ver[id][0];
        c_ver[1] = const_ver[id][1];
        c_ver[2] = const_ver[id][2];
        c_ver[3] = 1.0f;

        vec4_t d_light;
        d_light[0] = (*light_pos)[0];
        d_light[1] = (*light_pos)[1];
        d_light[2] = (*light_pos)[2];
        d_light[3] = 1.0f;
        vec4_multiply_mat(*v, d_light, d_light);

        vec4_t d_ver, d_half;
        vec4_multiply_mat(*vm, c_ver, d_ver);
        vec4_subtract(d_light, d_ver, d_light);
        vec4_subtract(d_light, d_ver, d_half);

        vec3_t light, half;
        light[0] = d_light[0];
        light[1] = d_light[1];
        light[2] = d_light[2];
        half[0] = d_half[0];
        half[1] = d_half[1];
        half[2] = d_half[2];
        vec3_normalize(light);
        vec3_normalize(half);

        ver[id].light[0] = light[0];
        ver[id].light[1] = light[1];
        ver[id].light[2] = light[2];
        ver[id].half[0] = half[0];
        ver[id].half[1] = half[1];
        ver[id].half[2] = half[2];

        vec4_t p_ver;
        vec4_multiply_mat(*pvm, c_ver, p_ver);
        float div = 1.0f / p_ver[3];
        ver[id].position[0] = p_ver[0] * div;
        ver[id].position[1] = p_ver[1] * div;
        ver[id].position[2] = p_ver[2] * div;
    }
} /* vertex_shader_vertices_kernel */


/* Transform normals */
__global__ static void vertex_shader_normals_kernel(
    mat4_t * __restrict__ vm,        // Inverse view model transformation matrix
    size_t s_nor,                    // Size of normal buffer
    vec3_t * __restrict__ const_nor, // Normal buffer
    vec3_t * __restrict__ nor        // Target normal buffer
) {
    size_t id = (size_t)(blockIdx.x * blockDim.x + threadIdx.x);
    if (id < s_nor) {
        vec4_t v;
        v[0] = const_nor[id][0];
        v[1] = const_nor[id][1];
        v[2] = const_nor[id][2];
        v[3] = 0.0f;
        vec4_multiply_mat(*vm, v, v);

        vec3_t n;
        n[0] = v[0];
        n[1] = v[1];
        n[2] = v[2];
        vec3_normalize(n);

        nor[id][0] = n[0];
        nor[id][1] = n[1];
        nor[id][2] = n[2];
    }
} /* vertex_shader_vertices_kernel */


/* Get ---------------------------------------------------------------------- */


/* Get size of vertex buffer */
extern inline size_t vertex_get_size_vertices(
    void
) {
    return device_size_vertices;
} /* vertex_get_size_vertices */


/* Get size of normal buffer */
extern inline size_t vertex_get_size_normals(
    void
) {
    return device_size_normals;
} /* vertex_get_size_normals */


/* Get vertex buffer */
extern inline vertex_t *vertex_get_vertices(
    void
) {
    return device_vertices;
} /* vertex_get_vertices */


/* Get normal buffer */
extern inline vec3_t *vertex_get_normals(
    void
) {
    return device_normals;
} /* vertex_get_normals */


/* Get light position */
extern inline vec3_t *vertex_get_light_position(
    void
) {
    return &host_light_position;
} /* vertex_get_light_position */


/* Get view transformation matrix */
extern inline mat4_t *vertex_get_view(
    void
) {
    return &host_view;
} /* vertex_get_view */


/* Get view model transformation matrix */
extern inline mat4_t *vertex_get_view_model(
    void
) {
    return &host_view_model;
} /* vertex_get_view_model */


/* Get inverse transpose view model transformation matrix */
extern inline mat4_t *vertex_get_view_model_inverse_transpose(
    void
) {
    return &host_view_model_inverse_transpose;
} /* vertex_get_view_model_inverse_transpose */


/* Get perspective view model transformation matrix */
extern inline mat4_t *vertex_get_perspective_view_model(
    void
) {
    return &host_perspective_view_model;
} /* vertex_get_perspective_view_model */


/* Verbose ------------------------------------------------------------------ */


#ifdef _DEBUG


/* Verbose output */
extern void vertex_verbose(
    long long time // Elapsed time
) {
    LOG_TEXT("Vertex shader");
    LOG_TRACE;
    print(PRINT_LIME_BG, " %lld ", time);
    print(PRINT_GREEN_BG, " us \n");
    print(PRINT_CYAN_BG, " Matrix \n");
    print(PRINT_AQUA_BG, " View model \n");
    verbose_mat4(host_view_model);
    print(PRINT_AQUA_BG, " Inverse transpose view model \n");
    verbose_mat4(host_view_model_inverse_transpose);
    print(PRINT_AQUA_BG, " Perspective view model \n");
    verbose_mat4(host_perspective_view_model);

    vertex_t *reference_vertices = (vertex_t *)malloc(
        device_size_vertices * sizeof(vertex_t));
    vec3_t *reference_normals = (vec3_t *)malloc(
        device_size_normals * sizeof(vec3_t));

    cudaMemcpy(
        reference_vertices,
        device_vertices,
        device_size_vertices * sizeof(vertex_t),
        cudaMemcpyDeviceToHost
    );
    cudaMemcpy(
        reference_normals,
        device_normals,
        device_size_normals * sizeof(vec3_t),
        cudaMemcpyDeviceToHost
    );

    print(PRINT_CYAN_BG, " Buffer \n");
    print(PRINT_AQUA_BG, " %llu vertices \n", device_size_vertices);
    verbose_vertex_array(device_size_vertices, reference_vertices);
    print(PRINT_AQUA_BG, " %llu normals \n", device_size_normals);
    verbose_vec3_array(device_size_normals, reference_normals);

    free(reference_vertices);
    free(reference_normals);
} /* vertex_verbose */


#endif /* _DEBUG */


/* -------------------------------------------------------------------------- */


/* vertex.cu */