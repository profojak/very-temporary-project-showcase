/* CUDA C Software Rendering Pipeline ---------------------------------------
   Bachelor Thesis                                              Jakub Profota
   Summer Semester 2022/3                    OI - Computer Games and Graphics
   Supervisor Jiri Bittner               Czech Technical University in Prague
   scene.c ------------------------------------------------------------------ */


#include "scene.h"

#include "define.h"
#include "verbose.h"
#include "../platform/memory.h"
#include "../platform/mutex.h"
#include "../platform/print.h"

#include <stdlib.h> // EXIT_SUCCESS, EXIT_FAILURE


// Model position
#define SCENE_MODEL_POSITION_DEFAULT 0.0f // Default position

// Model scale
#define SCENE_MODEL_SCALE_DEFAULT 1.0f // Default scale

// Model rotation
#define SCENE_MODEl_ROTATION_DEFAULT 0.0f // Default rotation
#define SCENE_MODEl_ROTATION_MAX 360.0f   // Maximum rotation
#define SCENE_MODEl_ROTATION_MIN 0.0f     // Minimum rotation

// Camera vertical field of view
#define SCENE_CAM_FOV_DEFAULT 80.0f // Default vertical field of view
#define SCENE_CAM_FOV_MAX 179.0f    // Maximum vertical field of view
#define SCENE_CAM_FOV_MIN 1.0f      // Minimum vertical field of view

// Camera clipping planes
#define SCENE_CAM_CLIP_NEAR_DEFAULT 0.01f // Default near clipping plane
#define SCENE_CAM_CLIP_FAR_DEFAULT 10.0f  // Default far clipping plane

// Camera configuration
#define SCENE_CAM_LONGITUDE_DEFAULT 75.0f // Default longitude
#define SCENE_CAM_LONGITUDE_MAX 360.0f    // Maximum longitude
#define SCENE_CAM_LONGITUDE_MIN 0.0f      // Minimum longitude
#define SCENE_CAM_LATITUDE_DEFAULT 10.0f  // Default latitude
#define SCENE_CAM_LATITUDE_MAX 89.9f      // Maximum latitude
#define SCENE_CAM_LATITUDE_MIN -89.9f     // Minimum latitude
#define SCENE_CAM_DISTANCE_DEFAULT 2.0f   // Default distance from origin
#define SCENE_CAM_DISTANCE_MIN 0.001f     // Minimum distance from origin

// Light position
#define SCENE_LIGHT_LONGITUDE_DEFAULT 45.0f // Default longitude
#define SCENE_LIGHT_LONGITUDE_MAX 360.0f    // Maximum longitude
#define SCENE_LIGHT_LONGITUDE_MIN 0.0f      // Minimum longitude
#define SCENE_LIGHT_LATITUDE_DEFAULT 45.0f  // Default latitude
#define SCENE_LIGHT_LATITUDE_MAX 360.0f     // Maximum latitude
#define SCENE_LIGHT_LATITUDE_MIN 0.0f       // Minimum latitude
#define SCENE_LIGHT_DISTANCE_DEFAULT 5.0f   // Default distance
#define SCENE_LIGHT_DISTANCE_MIN 0.0f       // Minimum distance

// Light constants
#define SCENE_LIGHT_CONSTANT_AMBIENT 0.2f  // Ambient constant
#define SCENE_LIGHT_CONSTANT_DIFFUSE 0.4f  // Diffuse constant
#define SCENE_LIGHT_CONSTANT_SPECULAR 0.4f // Specular constant

// Light colors of components
#define SCENE_LIGHT_COLOR_AMBIENT_R 255  // Ambient light red color
#define SCENE_LIGHT_COLOR_AMBIENT_G 255  // Ambient light green color
#define SCENE_LIGHT_COLOR_AMBIENT_B 255  // Ambient light blue color
#define SCENE_LIGHT_COLOR_DIFFUSE_R 255  // Diffuse light red color
#define SCENE_LIGHT_COLOR_DIFFUSE_G 255  // Diffuse light green color
#define SCENE_LIGHT_COLOR_DIFFUSE_B 255  // Diffuse light blue color
#define SCENE_LIGHT_COLOR_SPECULAR_R 255 // Specular light red color
#define SCENE_LIGHT_COLOR_SPECULAR_G 255 // Specular light green color
#define SCENE_LIGHT_COLOR_SPECULAR_B 255 // Specular light blue color

// Light shininess
#define SCENE_LIGHT_SHININESS_DEFAULT 10 // Default shininess
#define SCENE_LIGHT_SHININESS_MIN 1 // Default shininess


static scene_t scene = { 0 };
static mutex_t mutex = { 0 };

static const vec3_t vec3_zero = { 0.0f, 0.0f, 0.0f };
static const vec3_t vec3_up = { 0.0f, 0.0f, 1.0f };

static mat4_t model;
static mat4_t view;
static mat4_t perspective;
static vec3_t *vertex_light_position;
static mat4_t *vertex_view;
static mat4_t *vertex_view_model;
static mat4_t *vertex_view_model_inverse_transpose;
static mat4_t *vertex_perspective_view_model;
static int *pixel_light_shininess;


/* Init and free ------------------------------------------------------------ */


/* Initialize scene */
extern int scene_init(
    int *device_light_shininess,                 // Light shininess
    vec3_t *device_light_position,               // Light position
    mat4_t *device_view,                         // View transformation matrix
    mat4_t *device_view_model,                   // View model transformation
    mat4_t *device_view_model_inverse_transpose, // Inverse transpose view model
    mat4_t *device_perspective_view_model        // Perspective view model
) {
    pixel_light_shininess = device_light_shininess;
    vertex_light_position = device_light_position;
    vertex_view = device_view;
    vertex_view_model = device_view_model;
    vertex_view_model_inverse_transpose = device_view_model_inverse_transpose;
    vertex_perspective_view_model = device_perspective_view_model;

    mutex_create(&mutex);

    // Model
    if (mesh_load(&scene.model) != EXIT_SUCCESS) {
        #ifdef _DEBUG
        LOG_TRACE;
        #endif /* _DEBUG */
        return EXIT_FAILURE;
    }
    scene.model_scale[0] = 1.0f;
    scene.model_scale[1] = 1.0f;
    scene.model_scale[2] = 1.0f;

    // Camera
    scene.camera_fov_y = SCENE_CAM_FOV_DEFAULT;
    scene.camera_clip_planes[0] = SCENE_CAM_CLIP_NEAR_DEFAULT;
    scene.camera_clip_planes[1] = SCENE_CAM_CLIP_FAR_DEFAULT;
    scene.camera_configuration[0] = SCENE_CAM_LONGITUDE_DEFAULT;
    scene.camera_configuration[1] = SCENE_CAM_LATITUDE_DEFAULT;
    scene.camera_configuration[2] = SCENE_CAM_DISTANCE_DEFAULT;

    // Light
    scene.light_configuration[0] = SCENE_LIGHT_LONGITUDE_DEFAULT;
    scene.light_configuration[1] = SCENE_LIGHT_LATITUDE_DEFAULT;
    scene.light_configuration[2] = SCENE_LIGHT_DISTANCE_DEFAULT;
    scene.light_constant[0] = SCENE_LIGHT_CONSTANT_AMBIENT;
    scene.light_constant[1] = SCENE_LIGHT_CONSTANT_DIFFUSE;
    scene.light_constant[2] = SCENE_LIGHT_CONSTANT_SPECULAR;
    scene.light_ambient.r = SCENE_LIGHT_COLOR_AMBIENT_R;
    scene.light_ambient.g = SCENE_LIGHT_COLOR_AMBIENT_G;
    scene.light_ambient.b = SCENE_LIGHT_COLOR_AMBIENT_B;
    scene.light_diffuse.r = SCENE_LIGHT_COLOR_DIFFUSE_R;
    scene.light_diffuse.g = SCENE_LIGHT_COLOR_DIFFUSE_G;
    scene.light_diffuse.b = SCENE_LIGHT_COLOR_DIFFUSE_B;
    scene.light_specular.r = SCENE_LIGHT_COLOR_SPECULAR_R;
    scene.light_specular.g = SCENE_LIGHT_COLOR_SPECULAR_G;
    scene.light_specular.b = SCENE_LIGHT_COLOR_SPECULAR_B;
    scene.light_shininess = SCENE_LIGHT_SHININESS_DEFAULT;

    #ifdef _DEBUG
    LOG_TEXT("Initialized scene");
    LOG_TRACE;

    if (global_verbose) {
        verbose_scene(&scene);
    }
    #endif /* _DEBUG */

    return EXIT_SUCCESS;
} /* scene_init */


/* Free scene */
extern int scene_free(
    void
) {
    mutex_destroy(&mutex);

    if (mesh_unload(&scene.model) != EXIT_SUCCESS) {
        #ifdef _DEBUG
        LOG_TRACE;
        #endif /* _DEBUG */
        return EXIT_FAILURE;
    }

    #ifdef _DEBUG
    LOG_TEXT("Freed scene");
    LOG_TRACE;
    #endif /* _DEBUG */

    return EXIT_SUCCESS;
} /* scene_free */


/* Scene fetch -------------------------------------------------------------- */


/* Feed scene data to pipeline stages */
extern inline int scene_fetch(
    void
) {
    mutex_lock(&mutex);

    mem_set(model, 0, sizeof(mat4_t));
    mem_set(view, 0, sizeof(mat4_t));
    mem_set(perspective, 0, sizeof(mat4_t));
    mem_set(*vertex_view_model, 0, sizeof(mat4_t));
    mem_set(*vertex_perspective_view_model, 0, sizeof(mat4_t));

    mat3_t mat = { 0 };
    mat3_identity(mat);
    mat3_scale_xyz(scene.model_scale, mat);
    mat3_rotate_xyz(scene.model_rotation, mat);
    mat3_to_mat4(mat, model);
    mat4_translate_xyz(scene.model_position, model);

    vec3_t camera_position = { 0 }, light_position = { 0 };
    vec3_position(scene.camera_configuration, camera_position);
    vec3_position(scene.light_configuration, light_position);
    (*vertex_light_position)[0] = light_position[0];
    (*vertex_light_position)[1] = light_position[1];
    (*vertex_light_position)[2] = light_position[2];

    *pixel_light_shininess = scene.light_shininess;

    mat4_lookat(
        camera_position,
        (float *)vec3_zero,
        (float *)vec3_up,
        view
    );
    mat4_perspective(
        scene.camera_fov_y,
        WINDOW_ASPECT_RATIO,
        scene.camera_clip_planes,
        perspective
    );

    mat4_multiply(
        view,
        model,
        *vertex_view_model
    );

    mat4_copy(view, *vertex_view);
    mat4_copy(*vertex_view_model, *vertex_view_model_inverse_transpose);
    mat4_transpose(*vertex_view_model_inverse_transpose);
    mat4_inverse(*vertex_view_model_inverse_transpose);

    mat4_multiply(
        perspective,
        *vertex_view_model,
        *vertex_perspective_view_model
    );

    mutex_unlock(&mutex);

    return EXIT_SUCCESS;
} /* scene_fetch */


/* Set ---------------------------------------------------------------------- */


/* Set scene for benchmark */
extern void scene_set_benchmark(
    int benchmark // Benchmark
) {
    if (benchmark == BENCHMARK_IDLE) {
        // Nothing
    } else if (benchmark == BENCHMARK_360) {
        // Nothing
    } else if (benchmark == BENCHMARK_ZOOM) {
        // Zoom camera and move far clipping plane
        scene.camera_clip_planes[1] = 75.0f;
        scene.camera_configuration[2] = 0.05f;
    }
} /* scene_set_benchmark */


/* Set model position along x axis */
extern void scene_set_model_position_x(
    float delta // Value
) {
    mutex_lock(&mutex);
    if (delta == SCENE_VALUE_RESET) {
        scene.model_position[0] = SCENE_MODEL_POSITION_DEFAULT;
    } else {
        scene.model_position[0] += delta;
    }
    mutex_unlock(&mutex);

    #ifdef _DEBUG
    if (global_verbose) {
        LOG_DEBUG("Model position X set to       %4.2f",
            scene.model_position[0]);
    }
    #endif /* _DEBUG */
} /* scene_set_model_position_x */


/* Set model position along y axis */
extern void scene_set_model_position_y(
    float delta // Value
) {
    mutex_lock(&mutex);
    if (delta == SCENE_VALUE_RESET) {
        scene.model_position[1] = SCENE_MODEL_POSITION_DEFAULT;
    } else {
        scene.model_position[1] += delta;
    }
    mutex_unlock(&mutex);

    #ifdef _DEBUG
    if (global_verbose) {
        LOG_DEBUG("Model position Y set to       %4.2f",
            scene.model_position[1]);
    }
    #endif /* _DEBUG */
} /* scene_set_model_position_y */


/* Set model position along z axis */
extern void scene_set_model_position_z(
    float delta // Value
) {
    mutex_lock(&mutex);
    if (delta == SCENE_VALUE_RESET) {
        scene.model_position[2] = SCENE_MODEL_POSITION_DEFAULT;
    } else {
        scene.model_position[2] += delta;
    }
    mutex_unlock(&mutex);

    #ifdef _DEBUG
    if (global_verbose) {
        LOG_DEBUG("Model position Z set to       %4.2f",
            scene.model_position[2]);
    }
    #endif /* _DEBUG */
} /* scene_set_model_position_z */


/* Set model scale along x axis */
extern void scene_set_model_scale_x(
    float delta // Value
) {
    mutex_lock(&mutex);
    if (delta == SCENE_VALUE_RESET) {
        scene.model_scale[0] = SCENE_MODEL_SCALE_DEFAULT;
    } else {
        scene.model_scale[0] += delta;
    }
    mutex_unlock(&mutex);

    #ifdef _DEBUG
    if (global_verbose) {
        LOG_DEBUG("Model scale X set to          %4.2f",
            scene.model_scale[0]);
    }
    #endif /* _DEBUG */
} /* scene_set_model_scale_x */


/* Set model scale along y axis */
extern void scene_set_model_scale_y(
    float delta // Value
) {
    mutex_lock(&mutex);
    if (delta == SCENE_VALUE_RESET) {
        scene.model_scale[1] = SCENE_MODEL_SCALE_DEFAULT;
    } else {
        scene.model_scale[1] += delta;
    }
    mutex_unlock(&mutex);

    #ifdef _DEBUG
    if (global_verbose) {
        LOG_DEBUG("Model scale Y set to          %4.2f",
            scene.model_scale[1]);
    }
    #endif /* _DEBUG */
} /* scene_set_model_scale_y */


/* Set model scale along z axis */
extern void scene_set_model_scale_z(
    float delta // Value
) {
    mutex_lock(&mutex);
    if (delta == SCENE_VALUE_RESET) {
        scene.model_scale[2] = SCENE_MODEL_SCALE_DEFAULT;
    } else {
        scene.model_scale[2] += delta;
    }
    mutex_unlock(&mutex);

    #ifdef _DEBUG
    if (global_verbose) {
        LOG_DEBUG("Model scale Z set to          %4.2f",
            scene.model_scale[2]);
    }
    #endif /* _DEBUG */
} /* scene_set_model_scale_z */


/* Set model rotation along x axis */
extern void scene_set_model_rotate_x(
    float delta // Value
) {
    mutex_lock(&mutex);
    if (delta == SCENE_VALUE_RESET) {
        scene.model_rotation[0] = SCENE_MODEL_SCALE_DEFAULT;
    } else {
        scene.model_rotation[0] += delta;
        if (scene.model_rotation[0] < SCENE_MODEl_ROTATION_MIN) {
            scene.model_rotation[0] += SCENE_MODEl_ROTATION_MAX;
        } else if (scene.model_rotation[0] > SCENE_MODEl_ROTATION_MAX) {
            scene.model_rotation[0] -= SCENE_MODEl_ROTATION_MAX;
        }
    }
    mutex_unlock(&mutex);

    #ifdef _DEBUG
    if (global_verbose) {
        LOG_DEBUG("Model rotation X set to       %4.2f",
            scene.model_rotation[0]);
    }
    #endif /* _DEBUG */
} /* scene_set_model_rotate_x */


/* Set model rotation along y axis */
extern void scene_set_model_rotate_y(
    float delta // Value
) {
    mutex_lock(&mutex);
    if (delta == SCENE_VALUE_RESET) {
        scene.model_rotation[1] = SCENE_MODEL_SCALE_DEFAULT;
    } else {
        scene.model_rotation[1] += delta;
        if (scene.model_rotation[1] < SCENE_MODEl_ROTATION_MIN) {
            scene.model_rotation[1] += SCENE_MODEl_ROTATION_MAX;
        } else if (scene.model_rotation[1] > SCENE_MODEl_ROTATION_MAX) {
            scene.model_rotation[1] -= SCENE_MODEl_ROTATION_MAX;
        }
    }
    mutex_unlock(&mutex);

    #ifdef _DEBUG
    if (global_verbose) {
        LOG_DEBUG("Model rotation Y set to       %4.2f",
            scene.model_rotation[1]);
    }
    #endif /* _DEBUG */
} /* scene_set_model_rotate_y */


/* Set model rotation along z axis */
extern void scene_set_model_rotate_z(
    float delta // Value
) {
    mutex_lock(&mutex);
    if (delta == SCENE_VALUE_RESET) {
        scene.model_rotation[2] = SCENE_MODEL_SCALE_DEFAULT;
    } else {
        scene.model_rotation[2] += delta;
        if (scene.model_rotation[2] < SCENE_MODEl_ROTATION_MIN) {
            scene.model_rotation[2] += SCENE_MODEl_ROTATION_MAX;
        } else if (scene.model_rotation[2] > SCENE_MODEl_ROTATION_MAX) {
            scene.model_rotation[2] -= SCENE_MODEl_ROTATION_MAX;
        }
    }
    mutex_unlock(&mutex);

    #ifdef _DEBUG
    if (global_verbose) {
        LOG_DEBUG("Model rotation Z set to       %4.2f",
            scene.model_rotation[2]);
    }
    #endif /* _DEBUG */
} /* scene_set_model_rotate_z */


/* Set camera vertical field of view */
extern void scene_set_camera_fov(
    float delta // Value
) {
    mutex_lock(&mutex);
    if (delta == SCENE_VALUE_RESET) {
        scene.camera_fov_y = SCENE_CAM_FOV_DEFAULT;
    } else {
        scene.camera_fov_y += delta;
        if (scene.camera_fov_y < SCENE_CAM_FOV_MIN) {
            scene.camera_fov_y = SCENE_CAM_FOV_MIN;
        } else if (scene.camera_fov_y > SCENE_CAM_FOV_MAX) {
            scene.camera_fov_y = SCENE_CAM_FOV_MAX;
        }
    }
    mutex_unlock(&mutex);

    #ifdef _DEBUG
    if (global_verbose) {
        LOG_DEBUG("Camera vertical FOV set to    %4.2f",
            scene.camera_fov_y);
    }
    #endif /* _DEBUG */
} /* scene_set_camera_fov */


/* Set camera near clipping plane */
extern void scene_set_camera_clip_near(
    float delta // Value
) {
    mutex_lock(&mutex);
    if (delta == SCENE_VALUE_RESET) {
        scene.camera_clip_planes[0] = SCENE_CAM_CLIP_NEAR_DEFAULT;
    } else {
        scene.camera_clip_planes[0] *= delta;
    }
    mutex_unlock(&mutex);

    #ifdef _DEBUG
    if (global_verbose) {
        LOG_DEBUG("Near clipping plane set to    %4.2f",
            scene.camera_clip_planes[0]);
    }
    #endif /* _DEBUG */
} /* scene_set_camera_clip_near */


/* Set camera far clipping plane */
extern void scene_set_camera_clip_far(
    float delta // Value
) {
    mutex_lock(&mutex);
    if (delta == SCENE_VALUE_RESET) {
        scene.camera_clip_planes[1] = SCENE_CAM_CLIP_FAR_DEFAULT;
    } else {
        scene.camera_clip_planes[1] += delta;
    }
    mutex_unlock(&mutex);

    #ifdef _DEBUG
    if (global_verbose) {
        LOG_DEBUG("Far clipping plane set to     %4.2f",
            scene.camera_clip_planes[1]);
    }
    #endif /* _DEBUG */
} /* scene_set_camera_clip_far */


/* Set camera longitude */
extern void scene_set_camera_longitude(
    float delta // Value
) {
    mutex_lock(&mutex);
    scene.camera_configuration[0] += delta;
    if (scene.camera_configuration[0] < SCENE_CAM_LONGITUDE_MIN) {
        scene.camera_configuration[0] += SCENE_CAM_LONGITUDE_MAX;
    } else if (scene.camera_configuration[0] > SCENE_CAM_LONGITUDE_MAX) {
        scene.camera_configuration[0] -= SCENE_CAM_LONGITUDE_MAX;
    }
    mutex_unlock(&mutex);

    #ifdef _DEBUG
    if (global_verbose) {
        LOG_DEBUG("Camera longitude set to       %8.6f",
            scene.camera_configuration[0]);
    }
    #endif /* _DEBUG */
} /* scene_set_camera_longitude */


/* Set camera latitude */
extern void scene_set_camera_latitude(
    float delta // Value
) {
    mutex_lock(&mutex);
    scene.camera_configuration[1] += delta;
    if (scene.camera_configuration[1] < SCENE_CAM_LATITUDE_MIN) {
        scene.camera_configuration[1] = SCENE_CAM_LATITUDE_MIN;
    } else if (scene.camera_configuration[1] > SCENE_CAM_LATITUDE_MAX) {
        scene.camera_configuration[1] = SCENE_CAM_LATITUDE_MAX;
    }
    mutex_unlock(&mutex);

    #ifdef _DEBUG
    if (global_verbose) {
        LOG_DEBUG("Camera latitude set to        %8.6f",
            scene.camera_configuration[1]);
    }
    #endif /* _DEBUG */
} /* scene_set_camera_latitude */


/* Set camera distance */
extern void scene_set_camera_distance(
    float delta // Value
) {
    mutex_lock(&mutex);
    scene.camera_configuration[2] *= delta;
    if (scene.camera_configuration[2] < SCENE_CAM_DISTANCE_MIN) {
        scene.camera_configuration[2] = SCENE_CAM_DISTANCE_MIN;
    }
    mutex_unlock(&mutex);

    #ifdef _DEBUG
    if (global_verbose) {
        LOG_DEBUG("Camera distance set to        %8.6f",
            scene.camera_configuration[2]);
    }
    #endif /* _DEBUG */
} /* scene_set_camera_distance */


/* Set light longitude */
extern void scene_set_light_longitude(
    float delta // Value
) {
    mutex_lock(&mutex);
    if (delta == SCENE_VALUE_RESET) {
        scene.light_configuration[0] = SCENE_LIGHT_LONGITUDE_DEFAULT;
    } else {
        scene.light_configuration[0] += delta;
        if (scene.light_configuration[0] < SCENE_LIGHT_LONGITUDE_MIN) {
            scene.light_configuration[0] += SCENE_LIGHT_LONGITUDE_MAX;
        } else if (scene.light_configuration[0] > SCENE_LIGHT_LONGITUDE_MAX) {
            scene.light_configuration[0] -= SCENE_LIGHT_LONGITUDE_MAX;
        }
    }
    mutex_unlock(&mutex);

    #ifdef _DEBUG
    if (global_verbose) {
        LOG_DEBUG("Light longitude set to        %4.2f",
            scene.light_configuration[0]);
    }
    #endif /* _DEBUG */
} /* scene_set_light_longitude */


/* Set light latitude */
extern void scene_set_light_latitude(
    float delta // Value
) {
    mutex_lock(&mutex);
    if (delta == SCENE_VALUE_RESET) {
        scene.light_configuration[1] = SCENE_LIGHT_LATITUDE_DEFAULT;
    } else {
        scene.light_configuration[1] += delta;
        if (scene.light_configuration[1] < SCENE_LIGHT_LATITUDE_MIN) {
            scene.light_configuration[1] += SCENE_LIGHT_LATITUDE_MAX;
        } else if (scene.light_configuration[1] > SCENE_LIGHT_LATITUDE_MAX) {
            scene.light_configuration[1] -= SCENE_LIGHT_LATITUDE_MAX;
        }
    }
    mutex_unlock(&mutex);

    #ifdef _DEBUG
    if (global_verbose) {
        LOG_DEBUG("Light latitude set to         %4.2f",
            scene.light_configuration[1]);
    }
    #endif /* _DEBUG */
} /* scene_set_light_latitude */


/* Set light distance from origin */
extern void scene_set_light_distance(
    float delta // Value
) {
    mutex_lock(&mutex);
    if (delta == SCENE_VALUE_RESET) {
        scene.light_configuration[2] = SCENE_LIGHT_DISTANCE_DEFAULT;
    } else {
        scene.light_configuration[2] += delta;
        if (scene.light_configuration[2] < SCENE_LIGHT_DISTANCE_MIN) {
            scene.light_configuration[2] = SCENE_LIGHT_DISTANCE_MIN;
        }
    }
    mutex_unlock(&mutex);

    #ifdef _DEBUG
    if (global_verbose) {
        LOG_DEBUG("Light distance set to         %4.2f",
            scene.light_configuration[2]);
    }
    #endif /* _DEBUG */
} /* scene_set_light_distance */


/* Set light shininess */
extern void scene_set_light_shininess(
    int delta // Value
) {
    mutex_lock(&mutex);
    scene.light_shininess += delta;
    if (scene.light_shininess < SCENE_LIGHT_SHININESS_MIN) {
        scene.light_shininess = SCENE_LIGHT_SHININESS_MIN;
    }
    mutex_unlock(&mutex);

    #ifdef _DEBUG
    if (global_verbose) {
        LOG_DEBUG("Light shininess set to        %4i",
            scene.light_shininess);
    }
    #endif /* _DEBUG */
} /* scene_set_light_shininess */



/* Get ---------------------------------------------------------------------- */


/* Get mesh */
extern inline mesh_t *scene_get_mesh(
    void
) {
    return &scene.model;
} /* scene_get_mesh */


/* Get light constants */
extern inline vec3_t *scene_get_light_constant(
    void
) {
    return &scene.light_constant;
} /* scene_get_light_constant */


/* Get light ambient color */
extern inline rgb_t *scene_get_light_ambient(
    void
) {
    return &scene.light_ambient;
} /* scene_get_light_ambient */


/* Get light diffuse color */
extern inline rgb_t *scene_get_light_diffuse(
    void
) {
    return &scene.light_diffuse;
} /* scene_get_light_diffuse */


/* Get light specular color */
extern inline rgb_t *scene_get_light_specular(
    void
) {
    return &scene.light_specular;
} /* scene_get_light_specular */


/* Verbose ------------------------------------------------------------------ */


#ifdef _DEBUG


/* Verbose output */
extern void scene_verbose(
    long long time // Elapsed time
) {
    LOG_TEXT("Scene fetch");
    LOG_TRACE;
    print(PRINT_LIME_BG, " %lld ", time);
    print(PRINT_GREEN_BG, " us \n");
    verbose_scene(&scene);
    print(PRINT_CYAN_BG, " Matrix \n");
    print(PRINT_AQUA_BG, " Model \n");
    verbose_mat4(model);
    print(PRINT_AQUA_BG, " View \n");
    verbose_mat4(view);
    print(PRINT_AQUA_BG, " Perspective \n");
    verbose_mat4(perspective);
} /* scene_verbose */


#endif /* _DEBUG */


/* -------------------------------------------------------------------------- */


/* scene.c */