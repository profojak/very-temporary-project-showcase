/* CUDA C Software Rendering Pipeline ---------------------------------------
   Bachelor Thesis                                              Jakub Profota
   Summer Semester 2022/3                    OI - Computer Games and Graphics
   Supervisor Jiri Bittner               Czech Technical University in Prague
   pixel.cu ----------------------------------------------------------------- */


#include "pixel.h"

#include "../core/define.h"
#include "../core/verbose.h"
#include "../platform/print.h"


static fragment_t *device_fragments;
static unsigned int *device_framebuffer;
static int host_light_shininess;
static vec3_t *device_light_constant;
static rgb_t *device_light_ambient;
static rgb_t *device_light_diffuse;
static rgb_t *device_light_specular;


__global__ static void pixel_shader_kernel(
    fragment_t *, unsigned int *, vec3_t *,
    rgb_t *, rgb_t *, rgb_t *, int
    #ifdef _DEBUG
    , int);
    #else
    );
    #endif /* _DEBUG */


/* Init and free ------------------------------------------------------------ */


/* Initialize pixel shader */
extern int pixel_init(
    unsigned int *framebuffer,                 // Framebuffer
    vec3_t * __restrict__ host_light_constant, // Host light constants
    rgb_t * __restrict__ host_light_ambient,   // Host light ambient color
    rgb_t * __restrict__ host_light_diffuse,   // Host light diffuse color
    rgb_t * __restrict__ host_light_specular   // Host light specular color
) {
    device_framebuffer = framebuffer;

    // Fragment
    if (cudaMalloc(
        &device_fragments, WINDOW_WIDTH * WINDOW_HEIGHT * sizeof(fragment_t)
    ) != cudaSuccess) {
        #ifdef _DEBUG
        LOG_ERROR("Failed to allocate fragment buffer!");
        #endif /* _DEBUG */
        return EXIT_FAILURE;
    }

    // Light
    if (cudaMalloc(
        &device_light_constant, sizeof(vec3_t)
    ) != cudaSuccess) {
        #ifdef _DEBUG
        LOG_ERROR("Failed to allocate light constants!");
        #endif /* _DEBUG */
        return EXIT_FAILURE;
    }

    if (cudaMalloc(
        &device_light_ambient, sizeof(rgb_t)
    ) != cudaSuccess) {
        #ifdef _DEBUG
        LOG_ERROR("Failed to allocate light color!");
        #endif /* _DEBUG */
        return EXIT_FAILURE;
    }

    if (cudaMalloc(
        &device_light_diffuse, sizeof(rgb_t)
    ) != cudaSuccess) {
        #ifdef _DEBUG
        LOG_ERROR("Failed to allocate light color!");
        #endif /* _DEBUG */
        return EXIT_FAILURE;
    }

    if (cudaMalloc(
        &device_light_specular, sizeof(rgb_t)
    ) != cudaSuccess) {
        #ifdef _DEBUG
        LOG_ERROR("Failed to allocate light color!");
        #endif /* _DEBUG */
        return EXIT_FAILURE;
    }

    // Copy light data
    if (cudaMemcpy(
        device_light_constant,
        host_light_constant,
        sizeof(vec3_t),
        cudaMemcpyHostToDevice
    ) != cudaSuccess) {
        #ifdef _DEBUG
        LOG_ERROR("Failed to copy light constants from host to device!");
        #endif /* _DEBUG */
        return EXIT_FAILURE;
    }

    if (cudaMemcpy(
        device_light_ambient,
        host_light_ambient,
        sizeof(rgb_t),
        cudaMemcpyHostToDevice
    ) != cudaSuccess) {
        #ifdef _DEBUG
        LOG_ERROR("Failed to copy light color from host to device!");
        #endif /* _DEBUG */
        return EXIT_FAILURE;
    }

    if (cudaMemcpy(
        device_light_diffuse,
        host_light_diffuse,
        sizeof(rgb_t),
        cudaMemcpyHostToDevice
    ) != cudaSuccess) {
        #ifdef _DEBUG
        LOG_ERROR("Failed to copy light color from host to device!");
        #endif /* _DEBUG */
        return EXIT_FAILURE;
    }

    if (cudaMemcpy(
        device_light_specular,
        host_light_specular,
        sizeof(rgb_t),
        cudaMemcpyHostToDevice
    ) != cudaSuccess) {
        #ifdef _DEBUG
        LOG_ERROR("Failed to copy light color from host to device!");
        #endif /* _DEBUG */
        return EXIT_FAILURE;
    }

    #ifdef _DEBUG
    LOG_TEXT("Initialized pixel shader");
    LOG_TRACE;
    #endif /* _DEBUG */

    return EXIT_SUCCESS;
} /* pixel_init */


/* Free pixel shader */
extern int pixel_free(
    void
) {
    device_framebuffer = 0;

    if (cudaFree(device_fragments) != cudaSuccess) {
        #ifdef _DEBUG
        LOG_ERROR("Failed to free fragment buffer!");
        #endif /* _DEBUG */
        return EXIT_FAILURE;
    }
    device_fragments = 0;

    if (cudaFree(device_light_constant) != cudaSuccess) {
        #ifdef _DEBUG
        LOG_ERROR("Failed to free light constants!");
        #endif /* _DEBUG */
        return EXIT_FAILURE;
    }
    device_light_constant = 0;

    if (cudaFree(device_light_ambient) != cudaSuccess) {
        #ifdef _DEBUG
        LOG_ERROR("Failed to free light ambient color!");
        #endif /* _DEBUG */
        return EXIT_FAILURE;
    }
    device_light_ambient = 0;

    if (cudaFree(device_light_diffuse) != cudaSuccess) {
        #ifdef _DEBUG
        LOG_ERROR("Failed to free light diffuse color!");
        #endif /* _DEBUG */
        return EXIT_FAILURE;
    }
    device_light_diffuse = 0;

    if (cudaFree(device_light_specular) != cudaSuccess) {
        #ifdef _DEBUG
        LOG_ERROR("Failed to free light specular color!");
        #endif /* _DEBUG */
        return EXIT_FAILURE;
    }
    device_light_specular = 0;

    #ifdef _DEBUG
    LOG_TEXT("Freed pixel shader");
    LOG_TRACE;
    #endif /* _DEBUG */

    return EXIT_SUCCESS;
} /* pixel_free */


/* Pixel shader ------------------------------------------------------------- */


/* Shade fragments to pixels */
extern inline int pixel_shader(
    void
) {
    dim3 blocks(WINDOW_WIDTH / 16, WINDOW_HEIGHT / 16);
    dim3 threads(16, 16);
    pixel_shader_kernel <<< blocks, threads >>> (
        device_fragments,
        device_framebuffer,
        device_light_constant,
        device_light_ambient,
        device_light_diffuse,
        device_light_specular,
        host_light_shininess
        #ifdef _DEBUG
        ,global_color
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
} /* pixel_shader */


/* Kernel ------------------------------------------------------------------- */


/* Shade fragments and pass them as pixels to framebuffer */
__global__ static void pixel_shader_kernel(
    fragment_t * __restrict__ fra,   // Fragment buffer
    unsigned int * __restrict__ buf, // Framebuffer
    vec3_t * __restrict__ const_con, // Light constants
    rgb_t * __restrict__ ambient,    // Ambient light color
    rgb_t * __restrict__ diffuse,    // Diffuse light color
    rgb_t * __restrict__ specular,   // Specular light color
    int shininess      // Light shininess
    #ifdef _DEBUG
    , int global_color // Coloring
    #endif /* _DEBUG */
) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    rgb_t color = { 206, 223, 245 };
    vec3_t con = { 0 };

    #ifdef _DEBUG
    if (global_color == COLOR_BLINN_PHONG) {
    #endif /* _DEBUG */
        con[0] = (*const_con)[0];
        con[1] = (*const_con)[1];
        con[2] = (*const_con)[2];
    #ifdef _DEBUG
    } else if (global_color == COLOR_AMBIENT) {
        con[0] = 1.0f;
        con[1] = 0.0f;
        con[2] = 0.0f;
    } else if (global_color == COLOR_DIFFUSE) {
        con[0] = 0.0f;
        con[1] = 1.0f;
        con[2] = 0.0f;
    } else if (global_color == COLOR_SPECULAR) {
        con[0] = 0.0f;
        con[1] = 0.0f;
        con[2] = 1.0f;
    }
    #endif /* _DEBUG */

    vec3_t normal;
    normal[0] = fra[x + y * WINDOW_WIDTH].normal[0];
    normal[1] = fra[x + y * WINDOW_WIDTH].normal[1];
    normal[2] = fra[x + y * WINDOW_WIDTH].normal[2];

    #ifdef _DEBUG
    if (global_color != COLOR_CAMERA && global_color != COLOR_NORMAL &&
        global_color != COLOR_DEPTH) {
    #endif /* _DEBUG */
        if (normal[0] || normal[1] || normal[2]) {
            vec3_t light;
            light[0] = fra[x + y * WINDOW_WIDTH].light[0];
            light[1] = fra[x + y * WINDOW_WIDTH].light[1];
            light[2] = fra[x + y * WINDOW_WIDTH].light[2];
            vec3_t half;
            half[0] = fra[x + y * WINDOW_WIDTH].half[0];
            half[1] = fra[x + y * WINDOW_WIDTH].half[1];
            half[2] = fra[x + y * WINDOW_WIDTH].half[2];

            float normal_light, normal_half;
            normal_light = vec3_dot(normal, light);
            normal_light = maximum(normal_light, 0.0f);
            normal_half = vec3_dot(normal, half);
            normal_half = maximum(normal_half, 0.0f);

            vec3_t fcolor = { 0 };
            fcolor[0] = ((float)ambient->r) * con[0];
            fcolor[1] = ((float)ambient->g) * con[0];
            fcolor[2] = ((float)ambient->b) * con[0];

            fcolor[0] += ((float)diffuse->r) * con[1] * normal_light;
            fcolor[1] += ((float)diffuse->g) * con[1] * normal_light;
            fcolor[2] += ((float)diffuse->b) * con[1] * normal_light;

            for (int i = 1; i < shininess; i++) {
                normal_half *= normal_half;
            }

            fcolor[0] += ((float)specular->r) * con[2] * normal_half;
            fcolor[1] += ((float)specular->g) * con[2] * normal_half;
            fcolor[2] += ((float)specular->b) * con[2] * normal_half;

            color.r = minimum(fcolor[0], 255.0f);
            color.g = minimum(fcolor[1], 255.0f);
            color.b = minimum(fcolor[2], 255.0f);
        }
    #ifdef _DEBUG
    } else if (global_color == COLOR_CAMERA) {
        vec3_t up = { 0.0f, 0.0f, 1.0f };
        float dot = vec3_dot(up, normal);
        dot = maximum(dot, 0.0f);
        color.r = (unsigned char)(dot * 255);
        color.g = (unsigned char)(dot * 255);
        color.b = (unsigned char)(dot * 255);
    } else if (global_color == COLOR_NORMAL) {
        color.r = (unsigned char)(normal[0] * 255);
        color.g = (unsigned char)(normal[1] * 255);
        color.b = (unsigned char)(normal[2] * 255);
    } else if (global_color == COLOR_DEPTH) {
        float depth = fra[x + y * WINDOW_WIDTH].depth;
        for (int i = 0; i < 6; i++) {
            depth *= depth;
        }
        color.r = (unsigned char)((1.0f - depth) * 255);
        color.g = (unsigned char)((1.0f - depth) * 255);
        color.b = (unsigned char)((1.0f - depth) * 255);
    }
    #endif /* _DEBUG */

    buf[x + y * WINDOW_WIDTH] = (unsigned int)color.b +
                                (((unsigned int)color.g) << 8) +
                                (((unsigned int)color.r) << 16);
} /* pixel_shader_kernel */


/* Get ---------------------------------------------------------------------- */


/* Get fragment buffer */
extern inline fragment_t *pixel_get_fragments(
    void
) {
    return device_fragments;
} /* pixel_get_fragments */


/* Get light shininess */
extern inline int *pixel_get_light_shininess(
    void
) {
    return &host_light_shininess;
} /* pixel_get_light_shininess */


/* Verbose ------------------------------------------------------------------ */


#ifdef _DEBUG


/* Verbose output */
extern void pixel_verbose(
    long long time // Elapsed time
) {
    LOG_TEXT("Pixel shader");
    LOG_TRACE;
    print(PRINT_LIME_BG, " %lld ", time);
    print(PRINT_GREEN_BG, " us \n");

    unsigned int *reference_framebuffer = (unsigned int *)malloc(
        WINDOW_WIDTH * WINDOW_HEIGHT * sizeof(unsigned int));

    cudaMemcpy(
        reference_framebuffer,
        device_framebuffer,
        WINDOW_WIDTH * WINDOW_HEIGHT * sizeof(unsigned int),
        cudaMemcpyDeviceToHost
    );

    print(PRINT_CYAN_BG, " Buffer \n");
    print(PRINT_AQUA_BG, " %i x %i pixels \n", WINDOW_WIDTH, WINDOW_HEIGHT);
    print(PRINT_WHITE_FG, " %06x", reference_framebuffer[0]);
    print(PRINT_GREEN_FG, ", ");
    print(PRINT_WHITE_FG, "%06x", reference_framebuffer[1]);
    print(PRINT_GREEN_FG, ", ");
    print(PRINT_WHITE_FG, "%06x", reference_framebuffer[2]);
    print(PRINT_GREEN_FG, ", ");
    print(PRINT_WHITE_FG, "%06x", reference_framebuffer[3]);
    print(PRINT_GREEN_FG, ", ");
    print(PRINT_WHITE_FG, "%06x", reference_framebuffer[4]);
    print(PRINT_GREEN_FG, ", ");
    print(PRINT_GRAY_FG, ". . .\n");
    print(PRINT_GRAY_FG, " . . .");
    print(PRINT_GREEN_FG, " , ");
    print(PRINT_WHITE_FG, "%06x",
        reference_framebuffer[WINDOW_WIDTH * WINDOW_HEIGHT - 5]);
    print(PRINT_GREEN_FG, ", ");
    print(PRINT_WHITE_FG, "%06x",
        reference_framebuffer[WINDOW_WIDTH * WINDOW_HEIGHT - 4]);
    print(PRINT_GREEN_FG, ", ");
    print(PRINT_WHITE_FG, "%06x",
        reference_framebuffer[WINDOW_WIDTH * WINDOW_HEIGHT - 3]);
    print(PRINT_GREEN_FG, ", ");
    print(PRINT_WHITE_FG, "%06x",
        reference_framebuffer[WINDOW_WIDTH * WINDOW_HEIGHT - 2]);
    print(PRINT_GREEN_FG, ", ");
    print(PRINT_WHITE_FG, "%06x\n",
        reference_framebuffer[WINDOW_WIDTH * WINDOW_HEIGHT - 1]);

    free(reference_framebuffer);
} /* pixel_verbose */


#endif /* _DEBUG */


/* -------------------------------------------------------------------------- */


/* pixel.cu */