/* CUDA C Software Rendering Pipeline ---------------------------------------
   Bachelor Thesis                                              Jakub Profota
   Summer Semester 2022/3                    OI - Computer Games and Graphics
   Supervisor Jiri Bittner               Czech Technical University in Prague
   rasterize.cu ------------------------------------------------------------- */


#include "rasterize.h"

#include "../core/define.h"
#include "../core/verbose.h"
#include "../platform/print.h"


#define PRIMITIVE_LINE_THRESHOLD 0.03f  // Line barycentric threshold
#define PRIMITIVE_POINT_THRESHOLD 0.95f // Point barycentric threshold


static size_t device_size_triangles;
static triangle_t *device_triangles;
static unsigned int *device_bins;
static unsigned int *device_tiles;
static fragment_t *device_fragments;


__global__ static void rasterization_bin_kernel(
    size_t, triangle_t *,
    #ifdef _DEBUG
    int,
    #endif /* _DEBUG */
    unsigned int *);
__global__ static void rasterization_tile_kernel(
    triangle_t *,
    #ifdef _DEBUG
    int,
    #endif /* _DEBUG */
    unsigned int *,
    #ifdef _DEBUG
    int,
    #endif /* _DEBUG */
    unsigned int *);
__global__ static void rasterization_rasterize_kernel(
    triangle_t *,
    #ifdef _DEBUG
    int,
    #endif /* _DEBUG */
    unsigned int *, fragment_t *
    #ifdef _DEBUG
    , int, int
    #endif /* _DEBUG */
    );


/* Init and free ------------------------------------------------------------ */


/* Initialize rasterization */
extern int rasterize_init(
    size_t size_triangles,               // Size of triangle buffer
    triangle_t * __restrict__ triangles, // Triangle buffer
    fragment_t * __restrict__ fragments  // Fragment buffer
) {
    device_size_triangles = size_triangles;
    device_triangles = triangles;
    device_fragments = fragments;

    // Bin
    if (cudaMalloc(
        &device_bins,
        RENDER_BIN_WIDTH * RENDER_BIN_HEIGHT *
        #ifdef _DEBUG
        global_bin
        #else
        RENDER_BIN_BUFFER
        #endif /* _DEBUG */
        * sizeof(unsigned int)
    ) != cudaSuccess) {
        #ifdef _DEBUG
        LOG_ERROR("Failed to allocate bin buffer!");
        #endif /* _DEBUG */
        return EXIT_FAILURE;
    }

    // Tile
    if (cudaMalloc(
        &device_tiles,
        RENDER_BIN_WIDTH * RENDER_BIN_SIZE *
        RENDER_BIN_HEIGHT * RENDER_BIN_SIZE *
        #ifdef _DEBUG
        global_tile
        #else
        RENDER_TILE_BUFFER
        #endif /* _DEBUG */
        * sizeof(unsigned int)
    ) != cudaSuccess) {
        #ifdef _DEBUG
        LOG_ERROR("Failed to allocate tile buffer!");
        #endif /* _DEBUG */
        return EXIT_FAILURE;
    }

    #ifdef _DEBUG
    LOG_TEXT("Initialized rasterization");
    LOG_TRACE;
    #endif /* _DEBUG */

    return EXIT_SUCCESS;
} /* rasterize_init */


/* Free rasterization */
extern int rasterize_free(
    void
) {
    device_size_triangles = 0;
    device_triangles = 0;
    device_fragments = 0;

    // Bin
    if (cudaFree(device_bins) != cudaSuccess) {
        #ifdef _DEBUG
        LOG_ERROR("Failed to free bin buffer!");
        #endif /* _DEBUG */
        return EXIT_FAILURE;
    }
    device_bins = 0;

    // Tile
    if (cudaFree(device_tiles) != cudaSuccess) {
        #ifdef _DEBUG
        LOG_ERROR("Failed to free tile buffer!");
        #endif /* _DEBUG */
        return EXIT_FAILURE;
    }
    device_tiles = 0;

    #ifdef _DEBUG
    LOG_TEXT("Freed rasterization");
    LOG_TRACE;
    #endif /* _DEBUG */

    return EXIT_SUCCESS;
} /* rasterize_free */


/* Rasterization ------------------------------------------------------------ */


/* Rasterize primitives to fragments */
extern inline int rasterization(
    void
) {
    if (cudaMemset(
        device_bins,
        0,
        RENDER_BIN_WIDTH * RENDER_BIN_HEIGHT *
        #ifdef _DEBUG
        global_bin
        #else
        RENDER_BIN_BUFFER
        #endif /* _DEBUG */
        * sizeof(unsigned int)
    ) != cudaSuccess) {
        #ifdef _DEBUG
        LOG_ERROR("Failed to zero out bin buffer!");
        #endif /* _DEBUG */
        return EXIT_FAILURE;
    }

    if (cudaMemset(
        device_tiles,
        0,
        RENDER_BIN_WIDTH * RENDER_BIN_SIZE *
        RENDER_BIN_HEIGHT * RENDER_BIN_SIZE *
        #ifdef _DEBUG
        global_tile
        #else
        RENDER_TILE_BUFFER
        #endif /* _DEBUG */
        * sizeof(unsigned int)
    ) != cudaSuccess) {
        #ifdef _DEBUG
        LOG_ERROR("Failed to zero out tile buffer!");
        #endif /* _DEBUG */
        return EXIT_FAILURE;
    }

    // Bin
    dim3 bin_blocks(RENDER_BIN_WIDTH, RENDER_BIN_HEIGHT);
    rasterization_bin_kernel <<< bin_blocks, RENDER_MAX_THREADS_BLOCK >>> (
        device_size_triangles,
        device_triangles,
        #ifdef _DEBUG
        global_bin,
        #endif /* _DEBUG */
        device_bins
    );

    #ifdef _DEBUG
    cudaError_t error = cudaPeekAtLastError();
    if (error != cudaSuccess) {
        LOG_ERROR("%s", cudaGetErrorString(error));
        return EXIT_FAILURE;
    }
    #endif /* _DEBUG */

    // Tile
    dim3 tile_blocks(RENDER_BIN_WIDTH * RENDER_BIN_SIZE,
                     RENDER_BIN_HEIGHT * RENDER_BIN_SIZE);
    rasterization_tile_kernel <<< tile_blocks, RENDER_MAX_THREADS_BLOCK >>> (
        device_triangles,
        #ifdef _DEBUG
        global_bin,
        #endif /* _DEBUG */
        device_bins,
        #ifdef _DEBUG
        global_tile,
        #endif /* _DEBUG */
        device_tiles
    );

    #ifdef _DEBUG
    error = cudaPeekAtLastError();
    if (error != cudaSuccess) {
        LOG_ERROR("%s", cudaGetErrorString(error));
        return EXIT_FAILURE;
    }
    #endif /* _DEBUG */

    // Fragment
    unsigned int threads = RENDER_TILE_SIZE * RENDER_TILE_SIZE;
    rasterization_rasterize_kernel <<< tile_blocks, threads >>> (
        device_triangles,
        #ifdef _DEBUG
        global_tile,
        #endif /* _DEBUG */
        device_tiles,
        device_fragments
        #ifdef _DEBUG
        , global_primitive,
        global_shading
        #endif /* _DEBUG */
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
} /* rasterization */


/* Kernel ------------------------------------------------------------------- */


/* Sort triangles to bins */
__global__ static void rasterization_bin_kernel(
    size_t s_tri,                  // Size of triangle buffer
    triangle_t * __restrict__ tri, // Triangle buffer
    #ifdef _DEBUG
    int s_bin,                     // Size of bin buffer
    #endif /* _DEBUG */
    unsigned int *bin              // Bin buffer
) {
    __shared__ unsigned int n_bin;
    n_bin = 0;
    __syncthreads();

    unsigned int bin_block = blockIdx.y * RENDER_BIN_WIDTH + blockIdx.x;

    float x_min = (float)(blockIdx.x *
                          RENDER_BIN_SIZE * RENDER_TILE_SIZE);
    float y_min = (float)(blockIdx.y *
                          RENDER_BIN_SIZE * RENDER_TILE_SIZE);
    float x_max = (float)((blockIdx.x + 1) *
                           RENDER_BIN_SIZE * RENDER_TILE_SIZE);
    float y_max = (float)((blockIdx.y + 1) *
                           RENDER_BIN_SIZE * RENDER_TILE_SIZE);

    unsigned int s_batch = s_tri / RENDER_MAX_THREADS_BLOCK + 1;
    unsigned int n_batch = 0;

    while (n_batch < s_batch) {
        unsigned int tri_id = threadIdx.x * s_batch + n_batch;
        if (tri_id < s_tri) {
            if (!tri[tri_id].discard[0] &&
                !tri[tri_id].discard[1]
                #ifdef _DEBUG
                && !tri[tri_id].discard[2]
                #endif /* _DEBUG */
            ) {
                vec2_t v_min, v_max;
                triangle_aabb_2d(tri[tri_id], v_min, v_max);
                if (v_min[0] > x_max || v_min[1] > y_max ||
                    v_max[0] < x_min || v_max[1] < y_min ) {
                } else {
                    int i = atomicAdd(&n_bin, 1);
                    #ifdef _DEBUG
                    if (i < s_bin - 2) {
                        bin[bin_block * s_bin + i + 1] = tri_id;
                    }
                    #else
                    if (i < RENDER_BIN_BUFFER - 2) {
                        bin[bin_block * RENDER_BIN_BUFFER + i + 1] = tri_id;
                    }
                    #endif /* _DEBUG */
                }
            }
        }
        n_batch++;
    }

    __syncthreads();
    #ifdef _DEBUG
    bin[bin_block * s_bin] = n_bin;
    #else
    bin[bin_block * RENDER_BIN_BUFFER] = n_bin;
    #endif /* _DEBUG */
} /* rasterization_bin_kernel */


/* Sort triangles to tiles */
__global__ static void rasterization_tile_kernel(
    triangle_t * __restrict__ tri,   // Triangle buffer
    #ifdef _DEBUG
    int s_bin,                       // Size of bin buffer
    #endif /* _DEBUG */
    unsigned int * __restrict__ bin, // Bin buffer
    #ifdef _DEBUG
    int s_tile,                      // Size of tile buffer
    #endif /* _DEBUG */
    unsigned int *tile               // Tile buffer
) {
    __shared__ unsigned int n_tile;
    n_tile = 0;
    __syncthreads();

    unsigned int tile_block = blockIdx.y * gridDim.x + blockIdx.x;
    unsigned int bin_block = tile_block / (RENDER_BIN_SIZE * RENDER_BIN_SIZE);
    unsigned int tile_id = tile_block % (RENDER_BIN_SIZE * RENDER_BIN_SIZE);

    float x_min = (float)((bin_block % RENDER_BIN_WIDTH) *
                  RENDER_BIN_SIZE * RENDER_TILE_SIZE) +
                  (tile_id % RENDER_BIN_SIZE) * RENDER_TILE_SIZE;
    float y_min = (float)((bin_block / RENDER_BIN_WIDTH) *
                  RENDER_BIN_SIZE * RENDER_TILE_SIZE) +
                  (tile_id / RENDER_BIN_SIZE) * RENDER_TILE_SIZE;
    float x_max = (float)((bin_block % RENDER_BIN_WIDTH) *
                  RENDER_BIN_SIZE * RENDER_TILE_SIZE) +
                  (tile_id % RENDER_BIN_SIZE + 1) * RENDER_TILE_SIZE;
    float y_max = (float)((bin_block / RENDER_BIN_WIDTH) *
                  RENDER_BIN_SIZE * RENDER_TILE_SIZE) +
                  (tile_id / RENDER_BIN_SIZE + 1) * RENDER_TILE_SIZE;

    #ifdef _DEBUG
    unsigned int s_tri = bin[bin_block * s_bin];
    if (s_tri > s_bin - 2) {
        s_tri = s_bin - 1;
    }
    #else
    unsigned int s_tri = bin[bin_block * RENDER_BIN_BUFFER];
    if (s_tri > RENDER_BIN_BUFFER - 2) {
        s_tri = RENDER_BIN_BUFFER - 1;
    }
    #endif /* _DEBUG */

    unsigned int s_batch = s_tri / RENDER_MAX_THREADS_BLOCK + 1;
    unsigned int n_batch = 0;

    while (n_batch < s_batch) {
        unsigned int bin_tri_id = threadIdx.x * s_batch + n_batch;
        if (bin_tri_id < s_tri) {
            #ifdef _DEBUG
            unsigned int tri_id = bin[bin_block * s_bin + bin_tri_id + 1];
            #else
            unsigned int tri_id = bin[bin_block * RENDER_BIN_BUFFER +
                                      bin_tri_id + 1];
            #endif /* _DEBUG */
            vec2_t v_min, v_max;
            triangle_aabb_2d(tri[tri_id], v_min, v_max);
            if (v_min[0] > x_max || v_min[1] > y_max ||
                v_max[0] < x_min || v_max[1] < y_min ) {
            } else {
                int i = atomicAdd(&n_tile, 1);
                #ifdef _DEBUG
                if (i < s_tile - 2) {
                    tile[tile_block * s_tile + i + 1] = tri_id;
                }
                #else
                if (i < RENDER_TILE_BUFFER - 2) {
                    tile[tile_block * RENDER_TILE_BUFFER + i + 1] = tri_id;
                }
                #endif /* _DEBUG */
            }
        }
        n_batch++;
    }

    __syncthreads();
    #ifdef _DEBUG
    tile[tile_block * s_tile] = n_tile;
    #else
    tile[tile_block * RENDER_TILE_BUFFER] = n_tile;
    #endif /* _DEBUG */
} /* rasterization_tile_kernel */


/* Rasterize triangles to fragments */
__global__ static void rasterization_rasterize_kernel(
    triangle_t * __restrict__ tri,    // Triangle buffer
    #ifdef _DEBUG
    int s_tile,                       // Size of tile buffer
    #endif /* _DEBUG */
    unsigned int * __restrict__ tile, // Tile buffer
    fragment_t * __restrict__ fra     // Fragment buffer
    #ifdef _DEBUG
    , int global_primitive,           // Primitive draw type
    int global_shading                // Shading
    #endif /* _DEBUG */
) {
    unsigned int tile_block = blockIdx.y * gridDim.x + blockIdx.x;
    unsigned int bin_block = tile_block / (RENDER_BIN_SIZE * RENDER_BIN_SIZE);
    unsigned int tile_id = tile_block % (RENDER_BIN_SIZE * RENDER_BIN_SIZE);
    unsigned int fragment_id = threadIdx.y * RENDER_TILE_SIZE + threadIdx.x;

    unsigned int x = (float)((bin_block % RENDER_BIN_WIDTH) *
                     RENDER_BIN_SIZE * RENDER_TILE_SIZE) +
                     (tile_id % RENDER_BIN_SIZE) * RENDER_TILE_SIZE +
                     fragment_id % RENDER_TILE_SIZE;
    unsigned int y = (float)((bin_block / RENDER_BIN_WIDTH) *
                     RENDER_BIN_SIZE * RENDER_TILE_SIZE) +
                     (tile_id / RENDER_BIN_SIZE) * RENDER_TILE_SIZE +
                     fragment_id / RENDER_TILE_SIZE;
 
    #ifdef _DEBUG
    unsigned int s_tri = tile[tile_block * s_tile];
    if (s_tri > s_tile - 2) {
        s_tri = s_tile - 1;
    }
    #else
    unsigned int s_tri = tile[tile_block * RENDER_TILE_BUFFER];
    if (s_tri > RENDER_TILE_BUFFER - 2) {
        s_tri = RENDER_TILE_BUFFER - 1;
    }
    #endif /* _DEBUG */

    float min_depth = 1.0f;
    vec3_t normal = { 0.0f, 0.0f, 0.0f };
    vec3_t light = { 0.0f, 0.0f, 0.0f };
    vec3_t half = { 0.0f, 0.0f, 0.0f };

    for (unsigned int k = 0; k < s_tri; k++) {
        #ifdef _DEBUG
        unsigned int id = tile[tile_block * s_tile + k + 1];
        #else
        unsigned int id = tile[tile_block * RENDER_TILE_BUFFER + k + 1];
        #endif /* _DEBUG */
        int to_draw = 0;

        vec2_t point;
        point[0] = (float)x + 0.5;
        point[1] = (float)y + 0.5;

        vec3_t barycentric;

        if (cartesian_to_barycentric(
            tri[id].vertices[0].position,
            tri[id].vertices[1].position,
            tri[id].vertices[2].position,
            point, &barycentric)) {
            if (barycentric[0] >= 0.0f && barycentric[0] <= 1.0f &&
                barycentric[1] >= 0.0f && barycentric[1] <= 1.0f &&
                barycentric[2] >= 0.0f && barycentric[2] <= 1.0f) {
                #ifdef _DEBUG
                if (global_primitive == PRIMITIVE_TRIANGLE) {
                #endif /* _DEBUG */
                    to_draw = 1;
                #ifdef _DEBUG
                } else if (global_primitive == PRIMITIVE_LINE &&
                    (barycentric[0] <= PRIMITIVE_LINE_THRESHOLD ||
                    barycentric[1] <= PRIMITIVE_LINE_THRESHOLD ||
                    barycentric[2] <= PRIMITIVE_LINE_THRESHOLD)) {
                    to_draw = 1;
                } else if (global_primitive == PRIMITIVE_POINT &&
                    (barycentric[0] >= PRIMITIVE_POINT_THRESHOLD ||
                    barycentric[1] >= PRIMITIVE_POINT_THRESHOLD ||
                    barycentric[2] >= PRIMITIVE_POINT_THRESHOLD)) {
                    to_draw = 1;
                }
                #endif /* _DEBUG */
            }
        }

        if (to_draw) {
            float depth = tri[id].vertices[0].position[2] * barycentric[0] +
                          tri[id].vertices[1].position[2] * barycentric[1] +
                          tri[id].vertices[2].position[2] * barycentric[2];
            if (depth < min_depth) {
                min_depth = depth;

                #ifdef _DEBUG
                if (global_shading == SHADING_FLAT) {
                    barycentric[0] = 1.0f / 3.0f;
                    barycentric[1] = barycentric[0];
                    barycentric[2] = 1.0f - barycentric[0] - barycentric[1];
                }
                #endif /* _DEBUG */

                normal[0] = tri[id].normals[0][0] * barycentric[0] +
                            tri[id].normals[1][0] * barycentric[1] +
                            tri[id].normals[2][0] * barycentric[2];
                normal[1] = tri[id].normals[0][1] * barycentric[0] +
                            tri[id].normals[1][1] * barycentric[1] +
                            tri[id].normals[2][1] * barycentric[2];
                normal[2] = tri[id].normals[0][2] * barycentric[0] +
                            tri[id].normals[1][2] * barycentric[1] +
                            tri[id].normals[2][2] * barycentric[2];
                vec3_normalize(normal);

                light[0] = tri[id].vertices[0].light[0] * barycentric[0] +
                           tri[id].vertices[1].light[0] * barycentric[1] +
                           tri[id].vertices[2].light[0] * barycentric[2];
                light[1] = tri[id].vertices[0].light[1] * barycentric[0] +
                           tri[id].vertices[1].light[1] * barycentric[1] +
                           tri[id].vertices[2].light[1] * barycentric[2];
                light[2] = tri[id].vertices[0].light[2] * barycentric[0] +
                           tri[id].vertices[1].light[2] * barycentric[1] +
                           tri[id].vertices[2].light[2] * barycentric[2];
                vec3_normalize(light);

                half[0] = tri[id].vertices[0].half[0] * barycentric[0] +
                          tri[id].vertices[1].half[0] * barycentric[1] +
                          tri[id].vertices[2].half[0] * barycentric[2];
                half[1] = tri[id].vertices[0].half[1] * barycentric[0] +
                          tri[id].vertices[1].half[1] * barycentric[1] +
                          tri[id].vertices[2].half[1] * barycentric[2];
                half[2] = tri[id].vertices[0].half[2] * barycentric[0] +
                          tri[id].vertices[1].half[2] * barycentric[1] +
                          tri[id].vertices[2].half[2] * barycentric[2];
                vec3_normalize(half);
            }
        }
    }

    fra[x + y * WINDOW_WIDTH].normal[0] = normal[0];
    fra[x + y * WINDOW_WIDTH].normal[1] = normal[1];
    fra[x + y * WINDOW_WIDTH].normal[2] = normal[2];
    fra[x + y * WINDOW_WIDTH].light[0] = light[0];
    fra[x + y * WINDOW_WIDTH].light[1] = light[1];
    fra[x + y * WINDOW_WIDTH].light[2] = light[2];
    fra[x + y * WINDOW_WIDTH].half[0] = half[0];
    fra[x + y * WINDOW_WIDTH].half[1] = half[1];
    fra[x + y * WINDOW_WIDTH].half[2] = half[2];
    fra[x + y * WINDOW_WIDTH].depth = min_depth;
} /* rasterization_rasterize_kernel */


/* Verbose ------------------------------------------------------------------ */


#ifdef _DEBUG


/* Verbose output */
void rasterize_verbose(
    long long time // Elapsed time
) {
    LOG_TEXT("Rasterization");
    LOG_TRACE;
    print(PRINT_LIME_BG, " %lld ", time);
    print(PRINT_GREEN_BG, " us \n");

    unsigned int *reference_bins = (unsigned int *)malloc(
        RENDER_BIN_WIDTH * RENDER_BIN_HEIGHT *
        global_bin * sizeof(unsigned int));
    unsigned int *reference_tiles = (unsigned int *)malloc(
        RENDER_BIN_WIDTH * RENDER_BIN_SIZE *
        RENDER_BIN_HEIGHT * RENDER_BIN_SIZE *
        global_tile * sizeof(unsigned int));
    fragment_t *reference_fragments = (fragment_t *)malloc(
        WINDOW_WIDTH * WINDOW_HEIGHT * sizeof(fragment_t));

    cudaMemcpy(
        reference_bins,
        device_bins,
        RENDER_BIN_WIDTH * RENDER_BIN_HEIGHT *
        global_bin * sizeof(unsigned int),
        cudaMemcpyDeviceToHost
    );
    cudaMemcpy(
        reference_tiles,
        device_tiles,
        RENDER_BIN_WIDTH * RENDER_BIN_SIZE *
        RENDER_BIN_HEIGHT * RENDER_BIN_SIZE *
        global_tile * sizeof(unsigned int),
        cudaMemcpyDeviceToHost
    );
    cudaMemcpy(
        reference_fragments,
        device_fragments,
        WINDOW_WIDTH * WINDOW_HEIGHT * sizeof(fragment_t),
        cudaMemcpyDeviceToHost
    );

    size_t bins_assigned = 0;
    int bins_overflow = 0;
    int bins_max = 0, bins_max_id = 0;
    for (int i = 0; i < RENDER_BIN_WIDTH * RENDER_BIN_HEIGHT; i++) {
        bins_assigned += (size_t)reference_bins[global_bin * i];
        if (reference_bins[global_bin * i] > (unsigned int)global_bin) {
            bins_overflow++;
        }
        if (reference_bins[global_bin * i] > (unsigned int)bins_max) {
            bins_max = reference_bins[global_bin * i];
            bins_max_id = i;
        }
    }

    size_t tiles_assigned = 0;
    int tiles_overflow = 0;
    int tiles_max = 0, tiles_max_id = 0;
    for (int i = 0; i < RENDER_BIN_WIDTH * RENDER_BIN_SIZE *
                        RENDER_BIN_HEIGHT * RENDER_BIN_SIZE; i++) {
        tiles_assigned += (size_t)reference_tiles[global_tile * i];
        if (reference_tiles[global_tile * i] > (unsigned int)global_tile) {
            tiles_overflow++;
        }
        if (reference_tiles[global_tile * i] > (unsigned int)tiles_max) {
            tiles_max = reference_tiles[global_tile * i];
            tiles_max_id = i;
        }
    }

    print(PRINT_CYAN_BG, " Buffer \n");
    print(PRINT_AQUA_BG, " %i x %i bins \n",
        RENDER_BIN_WIDTH, RENDER_BIN_HEIGHT);
    if (bins_overflow) {
        LOG_WARNING("Bin overflow!");
    }
    print(PRINT_WHITE_FG, " %llu bin assignments\n", bins_assigned);
    print(PRINT_WHITE_FG, " %ix bin overflow, "
        "maximum of %i triangles in bin %i\n",
        bins_overflow, bins_max, bins_max_id + 1);
    for (int i = 0; i < RENDER_BIN_WIDTH * RENDER_BIN_HEIGHT; i++) {
        if (i == 0 || i == RENDER_BIN_WIDTH * RENDER_BIN_HEIGHT - 1 ||
            i == RENDER_BIN_WIDTH * RENDER_BIN_HEIGHT / 2 + 
                 RENDER_BIN_WIDTH / 2 - 1 ||
            i == RENDER_BIN_WIDTH * RENDER_BIN_HEIGHT / 2 +
                 RENDER_BIN_WIDTH / 2
        ) {
            print(PRINT_BLUE_BG, " %i \n", i + 1);
            print(PRINT_GRAY_FG, " triangles    indices\n");
            int k = global_bin * i;
            print(PRINT_WHITE_FG, " %9i", reference_bins[k]);
            print(PRINT_LIME_FG, ": ");
            for (int j = 1; j < 5; j++) {
                print(PRINT_WHITE_FG, "%9i", reference_bins[k + j]);
                print(PRINT_GREEN_FG, ", ");
            }
            print(PRINT_GRAY_FG, ". . .\n");
        } else if (i == 2 || i == RENDER_BIN_WIDTH * RENDER_BIN_HEIGHT - 2) {
            print(PRINT_GRAY_FG, " . . .\n");
        }
    }
    print(PRINT_AQUA_BG, " %i x %i tiles \n",
        RENDER_BIN_WIDTH * RENDER_BIN_SIZE,
        RENDER_BIN_HEIGHT * RENDER_BIN_SIZE);
    if (tiles_overflow) {
        LOG_WARNING("Tile overflow!");
    }
    print(PRINT_WHITE_FG, " %llu tile assignments\n", tiles_assigned);
    print(PRINT_WHITE_FG, " %ix tile overflow, "
        "maximum of %i triangles in tile %i\n",
        tiles_overflow, tiles_max, tiles_max_id + 1);
    for (int i = 0; i < RENDER_BIN_WIDTH * RENDER_BIN_SIZE *
                        RENDER_BIN_HEIGHT * RENDER_BIN_SIZE; i++) {
        if (i == 0 || i == RENDER_BIN_WIDTH * RENDER_BIN_SIZE *
                           RENDER_BIN_HEIGHT * RENDER_BIN_SIZE - 1 ||
            i == (RENDER_BIN_WIDTH * RENDER_BIN_HEIGHT / 2 +
                 RENDER_BIN_WIDTH / 2) *
                 RENDER_BIN_SIZE * RENDER_BIN_SIZE - 1 ||
            i == (RENDER_BIN_WIDTH * RENDER_BIN_HEIGHT / 2 +
                 RENDER_BIN_WIDTH / 2) *
                 RENDER_BIN_SIZE * RENDER_BIN_SIZE
        ) {
            print(PRINT_BLUE_BG, " %i \n", i + 1);
            print(PRINT_GRAY_FG, " triangles    indices\n");
            int k = global_tile * i;
            print(PRINT_WHITE_FG, " %9i", reference_tiles[k]);
            print(PRINT_LIME_FG, ": ");
            for (int j = 1; j < 5; j++) {
                print(PRINT_WHITE_FG, "%9i", reference_tiles[k + j]);
                print(PRINT_GREEN_FG, ", ");
            }
            print(PRINT_GRAY_FG, ". . .\n");
        } else if (i == 2 || i == RENDER_BIN_WIDTH * RENDER_BIN_SIZE *
                                  RENDER_BIN_HEIGHT * RENDER_BIN_SIZE - 2) {
            print(PRINT_GRAY_FG, " . . .\n");
        }
    }
    print(PRINT_AQUA_BG, " %i x %i fragments \n", WINDOW_WIDTH, WINDOW_HEIGHT);
    for (int i = 0; i < WINDOW_HEIGHT * WINDOW_WIDTH; i++) {
        if (i == 0 || i == WINDOW_WIDTH * WINDOW_HEIGHT - 1 ||
            i == WINDOW_WIDTH * WINDOW_HEIGHT / 2 - WINDOW_WIDTH / 2 - 1 ||
            i == WINDOW_WIDTH * WINDOW_HEIGHT / 2 - WINDOW_WIDTH / 2
        ) {
            print(PRINT_BLUE_BG, " %i \n", i + 1);
            print(PRINT_GRAY_FG, "            normal\n");
            verbose_vec3(reference_fragments[i].normal);
            print(PRINT_GRAY_FG, "             light\n");
            verbose_vec3(reference_fragments[i].light);
            print(PRINT_GRAY_FG, "              half\n");
            verbose_vec3(reference_fragments[i].half);
        } else if (i == 2 || i == WINDOW_WIDTH * WINDOW_HEIGHT - 2) {
            print(PRINT_GRAY_FG, " . . .\n");
        }
    }

    free(reference_bins);
    free(reference_fragments);
} /* rasterize_verbose */


#endif /* _DEBUG */


/* -------------------------------------------------------------------------- */


/* rasterize.cu */