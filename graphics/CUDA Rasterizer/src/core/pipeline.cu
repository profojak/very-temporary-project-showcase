/* CUDA C Software Rendering Pipeline ---------------------------------------
   Bachelor Thesis                                              Jakub Profota
   Summer Semester 2022/3                    OI - Computer Games and Graphics
   Supervisor Jiri Bittner               Czech Technical University in Prague
   pipeline.cu -------------------------------------------------------------- */


#include "define.h"
#include "scene.h"
#include "../platform/print.h"
#include "../platform/thread.h"
#include "../platform/timer.h"
#include "../platform/window.h"
#include "../render/pixel.h"
#include "../render/primitive.h"
#include "../render/rasterize.h"
#include "../render/rop.h"
#include "../render/vertex.h"

#include <stdio.h>  // FILE


#define GLOBAL_BIN_DEFAULT 256
#define GLOBAL_TILE_DEFAULT 256
#define GLOBAL_DEGENERATE_DEFAULT 1.0f


// Command line arguments and runtime globals
int global_align = 0;
int global_layout = 0;
char *global_mesh = 0;
int global_verbose = 0;
int global_benchmark = 0;
static int benchmark_backup = 0;
static int verbose_backup = 0;
#ifdef _DEBUG
int global_bin = GLOBAL_BIN_DEFAULT;
int global_tile = GLOBAL_TILE_DEFAULT;
float global_degenerate = GLOBAL_DEGENERATE_DEFAULT;
#endif /* _DEBUG */


static thread_t render_thread = { 0 };
static FILE *file = 0;


static int main_procedure(void);
static int render_procedure(void);

static int arguments(void *);


/* Entry -------------------------------------------------------------------- */


/* Entry point of software rendering pipeline */
extern "C" int pipeline(
    void *handle_instance, // Handle to identify executable
    void *args,            // String of command line arguments
    int show_flag          // Window flag to show or hide window
) {
    // Initialization
    if (print_init() != EXIT_SUCCESS) {
        #ifdef _DEBUG
        LOG_TRACE;
        #endif /* _DEBUG */
        return EXIT_FAILURE;
    }

    if (arguments(args) != EXIT_SUCCESS) {
        #ifdef _DEBUG
        LOG_TRACE;
        #endif /* _DEBUG */
        return EXIT_FAILURE;
    }

    if (timer_init() != EXIT_SUCCESS) {
        #ifdef _DEBUG
        LOG_TRACE;
        #endif /* _DEBUG */
        return EXIT_FAILURE;
    }

    if (window_init(handle_instance, show_flag) != EXIT_SUCCESS) {
        #ifdef _DEBUG
        LOG_TRACE;
        #endif /* _DEBUG */
        return EXIT_FAILURE;
    }

    if (thread_create(&render_thread, &render_procedure, 0) != EXIT_SUCCESS) {
        #ifdef _DEBUG
        LOG_TRACE;
        #endif /* _DEBUG */
        return EXIT_FAILURE;
    }

    // Procedure
    if (main_procedure() != EXIT_SUCCESS) {
        #ifdef _DEBUG
        LOG_TRACE;
        #endif /* _DEBUG */
        return EXIT_FAILURE;
    }

    // Cleanup
    if (thread_destroy(&render_thread) != EXIT_SUCCESS) {
        #ifdef _DEBUG
        LOG_TRACE;
        #endif /* _DEBUG */
        return EXIT_FAILURE;
    }

    #ifdef _DEBUG
    LOG_EMPTY;
    LOG_INFO("Main thread cleanup");
    #endif /* _DEBUG */

    if (window_free() != EXIT_SUCCESS) {
        #ifdef _DEBUG
        LOG_TRACE;
        #endif /* _DEBUG */
        return EXIT_FAILURE;
    }

    if (timer_free() != EXIT_SUCCESS) {
        #ifdef _DEBUG
        LOG_TRACE;
        #endif /* _DEBUG */
        return EXIT_FAILURE;
    }

    if (print_free() != EXIT_SUCCESS) {
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
} /* pipeline */


/* Procedure ---------------------------------------------------------------- */


/* Main procedure */
static int main_procedure(
    void
) {
    #ifdef _DEBUG
    LOG_TEXT("Entered main procedure");
    LOG_TRACE;
    #endif /* _DEBUG */

    while (window_get_status() == WINDOW_ACTIVE) {
        window_callback_message();
    }

    return EXIT_SUCCESS;
} /* main_procedure */


/* Render procedure */
static int render_procedure(
    void
) {
    #ifdef _DEBUG
    thread_sleep(10);

    LOG_EMPTY;
    LOG_INFO("Render thread initialization");
    LOG_TEXT("Entered render procedure");
    LOG_TRACE;
    #endif /* _DEBUG */

    const unsigned long long frame_time_target = 1000000 / WINDOW_FPS;
    unsigned int frame_counter = 0;
    timer_t frame_timer = { 0 };
    timer_t verbose_timer = { 0 };
    long long time_total, time_scene_fetch, time_vertex_shader;
    long long time_primitive_assembly, time_rasterization;
    long long time_pixel_shader, time_raster_operation;

    // Initialization
    if (scene_init(
        pixel_get_light_shininess(),
        vertex_get_light_position(),
        vertex_get_view(),
        vertex_get_view_model(),
        vertex_get_view_model_inverse_transpose(),
        vertex_get_perspective_view_model()
    ) != EXIT_SUCCESS) {
        #ifdef _DEBUG
        LOG_TRACE;
        #endif /* _DEBUG */
        return EXIT_FAILURE;
    }

    if (vertex_init(
        scene_get_mesh()->size_vertices,
        scene_get_mesh()->vertices,
        scene_get_mesh()->size_normals,
        scene_get_mesh()->normals
    ) != EXIT_SUCCESS) {
        #ifdef _DEBUG
        LOG_TRACE;
        #endif /* _DEBUG */
        return EXIT_FAILURE;
    }

    if (primitive_init(
        vertex_get_vertices(),
        vertex_get_normals(),
        scene_get_mesh()->size_indices,
        scene_get_mesh()->indices
    ) != EXIT_SUCCESS) {
        #ifdef _DEBUG
        LOG_TRACE;
        #endif /* _DEBUG */
        return EXIT_FAILURE;
    }

    if (rop_init() != EXIT_SUCCESS) {
        #ifdef _DEBUG
        LOG_TRACE;
        #endif /* _DEBUG */
        return EXIT_FAILURE;
    }

    if (pixel_init(
        rop_get_framebuffer(),
        scene_get_light_constant(),
        scene_get_light_ambient(),
        scene_get_light_diffuse(),
        scene_get_light_specular()
    ) != EXIT_SUCCESS) {
        #ifdef _DEBUG
        LOG_TRACE;
        #endif /* _DEBUG */
        return EXIT_FAILURE;
    }

    if (rasterize_init(
        primitive_get_size_triangles(),
        primitive_get_triangles(),
        pixel_get_fragments()
    ) != EXIT_SUCCESS) {
        #ifdef _DEBUG
        LOG_TRACE;
        #endif /* _DEBUG */
        return EXIT_FAILURE;
    }

    while (window_get_status() == WINDOW_ACTIVE) {
        // Benchmark
        if (global_benchmark > 0) {
            if (global_benchmark == BENCHMARK_IDLE) {
                print(PRINT_LIME_BG, " Idle benchmark \n");
                scene_set_benchmark(global_benchmark);
                file = fopen("benchmark_idle.txt", "w");
            } else if (global_benchmark == BENCHMARK_360) {
                print(PRINT_LIME_BG, " 360 degrees benchmark \n");
                scene_set_benchmark(global_benchmark);
                file = fopen("benchmark_360.txt", "w");
            } else if (global_benchmark == BENCHMARK_ZOOM) {
                print(PRINT_LIME_BG, " Zoom benchmark \n");
                scene_set_benchmark(global_benchmark);
                file = fopen("benchmark_zoom.txt", "w");
            }
            print(PRINT_WHITE_FG, " input is disabled\n");
            benchmark_backup = global_benchmark;
            verbose_backup = global_verbose;
            global_benchmark = -1;
            global_verbose = 0;
            frame_counter = 0;
            thread_sleep(500);
        }

        timer_update(&frame_timer);

        #ifdef _DEBUG
        if (global_verbose == VERBOSE_ACTIVE) {
            LOG_EMPTY;
            LOG_INFO("Verbose render pass");
        }
        #endif /* _DEBUG */
        timer_update(&verbose_timer);

        // Scene fetch
        if (scene_fetch() != EXIT_SUCCESS) {
            #ifdef _DEBUG
            LOG_TRACE;
            #endif /* _DEBUG */
            return EXIT_FAILURE;
        }

        timer_update(&verbose_timer);
        time_scene_fetch = verbose_timer.time_elapsed;
        #ifdef _DEBUG
        if (global_verbose == VERBOSE_ACTIVE) {
            scene_verbose(verbose_timer.time_elapsed);
        }
        timer_update(&verbose_timer);
        #endif /* _DEBUG */

        // Vertex shader
        if (vertex_shader() != EXIT_SUCCESS) {
            #ifdef _DEBUG
            LOG_TRACE;
            #endif /* _DEBUG */
            return EXIT_FAILURE;
        }

        timer_update(&verbose_timer);
        time_vertex_shader = verbose_timer.time_elapsed;
        #ifdef _DEBUG
        if (global_verbose == VERBOSE_ACTIVE) {
            vertex_verbose(verbose_timer.time_elapsed);
        }
        timer_update(&verbose_timer);
        #endif /* _DEBUG */

        // Primitive assembly
        if (primitive_assembly() != EXIT_SUCCESS) {
            #ifdef _DEBUG
            LOG_TRACE;
            #endif /* _DEBUG */
            return EXIT_FAILURE;
        }

        timer_update(&verbose_timer);
        time_primitive_assembly = verbose_timer.time_elapsed;
        #ifdef _DEBUG
        if (global_verbose == VERBOSE_ACTIVE) {
            primitive_verbose(verbose_timer.time_elapsed);
        }
        timer_update(&verbose_timer);
        #endif /* _DEBUG */

        // Rasterization
        if (rasterization() != EXIT_SUCCESS) {
            #ifdef _DEBUG
            LOG_TRACE;
            #endif /* _DEBUG */
            return EXIT_FAILURE;
        }

        timer_update(&verbose_timer);
        time_rasterization = verbose_timer.time_elapsed;
        #ifdef _DEBUG
        if (global_verbose == VERBOSE_ACTIVE) {
            rasterize_verbose(verbose_timer.time_elapsed);
        }
        timer_update(&verbose_timer);
        #endif /* _DEBUG */

        // Pixel shader
        if (pixel_shader() != EXIT_SUCCESS) {
            #ifdef _DEBUG
            LOG_TRACE;
            #endif /* _DEBUG */
            return EXIT_FAILURE;
        }

        timer_update(&verbose_timer);
        time_pixel_shader = verbose_timer.time_elapsed;
        #ifdef _DEBUG
        if (global_verbose == VERBOSE_ACTIVE) {
            pixel_verbose(verbose_timer.time_elapsed);
        }
        timer_update(&verbose_timer);
        #endif /* _DEBUG */

        // Raster operation
        if (raster_operation() != EXIT_SUCCESS) {
            #ifdef _DEBUG
            LOG_TRACE;
            #endif /* _DEBUG */
            return EXIT_FAILURE;
        }

        timer_update(&verbose_timer);
        time_raster_operation = verbose_timer.time_elapsed;
        #ifdef _DEBUG
        if (global_verbose == VERBOSE_ACTIVE) {
            rop_verbose(verbose_timer.time_elapsed);
        }
        #endif /* _DEBUG */

        // Sleep
        timer_update(&frame_timer);
        time_total = frame_timer.time_elapsed;

        // Benchmark
        if (benchmark_backup) {
            frame_counter++;
            if (frame_counter % WINDOW_FPS == 0) {
                print(PRINT_GRAY_FG, " *");
            }

            fprintf(file, "%u %lli %lli %lli %lli %lli %lli %lli\n",
                frame_counter, time_total,
                time_scene_fetch, time_vertex_shader,
                time_primitive_assembly, time_rasterization,
                time_pixel_shader, time_raster_operation
            );

            if (benchmark_backup == BENCHMARK_IDLE) {
                if (frame_counter == BENCHMARK_IDLE_FRAMES) {
                    global_benchmark = 0;
                    global_verbose = verbose_backup;
                    benchmark_backup = 0;
                    verbose_backup = 0;
                    fclose(file);
                    file = 0;
                    print(PRINT_WHITE_FG, "\n");
                    print(PRINT_LIME_BG, " Benchmark finished \n");
                    print(PRINT_WHITE_FG, " input is enabled\n");
                }
            } else if (benchmark_backup == BENCHMARK_360) {
                scene_set_camera_longitude(0.5f);
                if (frame_counter == BENCHMARK_360_FRAMES) {
                    global_benchmark = 0;
                    global_verbose = verbose_backup;
                    benchmark_backup = 0;
                    verbose_backup = 0;
                    fclose(file);
                    file = 0;
                    print(PRINT_WHITE_FG, "\n");
                    print(PRINT_LIME_BG, " Benchmark finished \n");
                    print(PRINT_WHITE_FG, " input is enabled\n");
                }
            } else if (benchmark_backup == BENCHMARK_ZOOM) {
                scene_set_camera_distance(1.01f);
                if (frame_counter == BENCHMARK_ZOOM_FRAMES) {
                    global_benchmark = 0;
                    global_verbose = verbose_backup;
                    benchmark_backup = 0;
                    verbose_backup = 0;
                    fclose(file);
                    file = 0;
                    print(PRINT_WHITE_FG, "\n");
                    print(PRINT_LIME_BG, " Benchmark finished \n");
                    print(PRINT_WHITE_FG, " input is enabled\n");
                }
            }
        }

        #ifndef _DEBUG
        if (global_verbose == VERBOSE_ACTIVE) {
            print(PRINT_WHITE_FG, "%lld us\n", frame_timer.time_elapsed);
        }
        #endif /* _DEBUG */

        if (frame_time_target > frame_timer.time_elapsed) {
            long long sleep = (long long)frame_time_target -
                frame_timer.time_elapsed - 1000;
            if (sleep > 0) {
                thread_sleep((unsigned int)(sleep / 1000));
            }
        }

        if (global_verbose == VERBOSE_ACTIVE) {
            global_verbose = VERBOSE_ENABLED;
        } else if (global_verbose == VERBOSE_QUEUED) {
            global_verbose = VERBOSE_ACTIVE;
        }
    }

    // Cleanup
    #ifdef _DEBUG
    LOG_EMPTY;
    LOG_INFO("Render thread cleanup");
    #endif /* _DEBUG */

    if (rasterize_free() != EXIT_SUCCESS) {
        #ifdef _DEBUG
        LOG_TRACE;
        #endif /* _DEBUG */
        return EXIT_FAILURE;
    }

    if (pixel_free() != EXIT_SUCCESS) {
        #ifdef _DEBUG
        LOG_TRACE;
        #endif /* _DEBUG */
        return EXIT_FAILURE;
    }

    if (rop_free() != EXIT_SUCCESS) {
        #ifdef _DEBUG
        LOG_TRACE;
        #endif /* _DEBUG */
        return EXIT_FAILURE;
    }

    if (primitive_free() != EXIT_SUCCESS) {
        #ifdef _DEBUG
        LOG_TRACE;
        #endif /* _DEBUG */
        return EXIT_FAILURE;
    }

    if (vertex_free() != EXIT_SUCCESS) {
        #ifdef _DEBUG
        LOG_TRACE;
        #endif /* _DEBUG */
        return EXIT_FAILURE;
    }

    if (scene_free() != EXIT_SUCCESS) {
        #ifdef _DEBUG
        LOG_TRACE;
        #endif /* _DEBUG */
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
} /* render_procedure */


/* Arguments ---------------------------------------------------------------- */


/* Process command line arguments */
static int arguments(
    void *args // Command line arguments
) {
    char *token = strtok((char *)args, " ");
    char *str = 0;
    while (token) {
        if (strcmp(token, "-align") == 0) {
            global_align = 1;
        } else if (strcmp(token, "-layout") == 0) {
            global_layout = 1;
        } else if (strcmp(token, "-verbose") == 0) {
            global_verbose = VERBOSE_ENABLED;
        } else if (str = strstr(token, "-model=")) {
            token = str;
            token += strlen("-model=");
            global_mesh = token;
        #ifdef _DEBUG
        } else if (str = strstr(token, "-bin=")) {
            token = str;
            token += strlen("-bin=");
            global_bin = atoi(token);
        } else if (str = strstr(token, "-tile=")) {
            token = str;
            token += strlen("-tile=");
            global_tile = atoi(token);
        } else if (str = strstr(token, "-degenerate=")) {
            token = str;
            token += strlen("-degenerate=");
            global_degenerate = (float)atof(token);
        } else {
            LOG_ERROR("Invalid command line argument \"%s\"!", token);
            return EXIT_FAILURE;
        #endif /* _DEBUG */
        }
        token = strtok(NULL, " ");
    }

    #ifdef _DEBUG
    LOG_TEXT("Processed arguments");
    LOG_TRACE;

    if (global_verbose) {
        LOG_DEBUG("-verbose");
        if (global_align) {
            LOG_DEBUG("-align");
        }
        if (global_mesh) {
            LOG_DEBUG("-model=%s", global_mesh);
        }
        if (global_bin != GLOBAL_BIN_DEFAULT) {
            LOG_DEBUG("-bin=%i", global_bin);
        }
        if (global_tile != GLOBAL_TILE_DEFAULT) {
            LOG_DEBUG("-tile=%i", global_tile);
        }
        if (global_degenerate != GLOBAL_DEGENERATE_DEFAULT) {
            LOG_DEBUG("-degenerate=%f", global_degenerate);
        }
    }
    #endif /* _DEBUG */

    return EXIT_SUCCESS;
} /* arguments */


/* -------------------------------------------------------------------------- */


/* pipeline.cu */