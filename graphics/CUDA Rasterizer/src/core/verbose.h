/* CUDA C Software Rendering Pipeline ---------------------------------------
   Bachelor Thesis                                              Jakub Profota
   Summer Semester 2022/3                    OI - Computer Games and Graphics
   Supervisor Jiri Bittner               Czech Technical University in Prague
   verbose.h ---------------------------------------------------------------- */


#ifndef VERBOSE_H
#define VERBOSE_H


#ifdef _DEBUG


#include "math.cuh"
#include "mesh.h"
#include "scene.h"


void verbose_vec2(vec2_t);
void verbose_ivec2(ivec2_t);
void verbose_vec3(vec3_t);
void verbose_ivec3(ivec3_t);
void verbose_vec4(vec4_t);
void verbose_ivec4(ivec4_t);

void verbose_vec2_array(size_t, vec2_t *);
void verbose_ivec2_array(size_t, ivec2_t *);
void verbose_vec3_array(size_t, vec3_t *);
void verbose_ivec3_array(size_t, ivec3_t *);
void verbose_vec4_array(size_t, vec4_t *);
void verbose_ivec4_array(size_t, ivec4_t *);

void verbose_mat3(mat3_t);
void verbose_mat4(mat4_t);

void verbose_rgb(rgb_t);
void verbose_vertex(vertex_t);
void verbose_vertex_array(size_t, vertex_t *);

void verbose_mesh(mesh_t *);

void verbose_scene(scene_t *);


/* -------------------------------------------------------------------------- */


#endif /* _DEBUG */


#endif /* VERBOSE_H */


/* verbose.h */