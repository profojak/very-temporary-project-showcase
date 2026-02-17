/* CUDA C Software Rendering Pipeline ---------------------------------------
   Bachelor Thesis                                              Jakub Profota
   Summer Semester 2022/3                    OI - Computer Games and Graphics
   Supervisor Jiri Bittner               Czech Technical University in Prague
   scene.h ------------------------------------------------------------------ */


#ifndef SCENE_H
#define SCENE_H


#include "math.cuh"
#include "mesh.h"


typedef struct scene_t {
    mesh_t model;                // Model mesh
    vec3_t model_position;       // Model position
    vec3_t model_scale;          // Model scale
    vec3_t model_rotation;       // Model rotation

    float camera_fov_y;          // Camera vertical field of view
    vec2_t camera_clip_planes;   // Near and far clipping planes
    vec3_t camera_configuration; // Camera configuration

    vec3_t light_configuration;  // Light configuration
    vec3_t light_constant;       // Ratios of reflection of light components
    rgb_t light_ambient;         // Color of ambient light component
    rgb_t light_diffuse;         // Color of diffuse light component
    rgb_t light_specular;        // Color of specular light component
    int light_shininess;         // Shininess constant of specular highlight
} scene_t;


int scene_init(int *, vec3_t *, mat4_t *, mat4_t *, mat4_t *, mat4_t *);
int scene_free(void);

int scene_fetch(void);

void scene_set_benchmark(int);
void scene_set_model_position_x(float);
void scene_set_model_position_y(float);
void scene_set_model_position_z(float);
void scene_set_model_scale_x(float);
void scene_set_model_scale_y(float);
void scene_set_model_scale_z(float);
void scene_set_model_rotate_x(float);
void scene_set_model_rotate_y(float);
void scene_set_model_rotate_z(float);
void scene_set_camera_fov(float);
void scene_set_camera_clip_near(float);
void scene_set_camera_clip_far(float);
void scene_set_camera_longitude(float);
void scene_set_camera_latitude(float);
void scene_set_camera_distance(float);
void scene_set_light_longitude(float);
void scene_set_light_latitude(float);
void scene_set_light_distance(float);
void scene_set_light_shininess(int);

mesh_t *scene_get_mesh(void);
vec3_t *scene_get_light_constant(void);
rgb_t *scene_get_light_ambient(void);
rgb_t *scene_get_light_diffuse(void);
rgb_t *scene_get_light_specular(void);

#ifdef _DEBUG
void scene_verbose(long long);
#endif /* _DEBUG */


/* -------------------------------------------------------------------------- */


#endif /* SCENE_H */


/* scene.h */