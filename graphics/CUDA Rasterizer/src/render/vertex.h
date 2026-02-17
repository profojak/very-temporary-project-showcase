/* CUDA C Software Rendering Pipeline ---------------------------------------
   Bachelor Thesis                                              Jakub Profota
   Summer Semester 2022/3                    OI - Computer Games and Graphics
   Supervisor Jiri Bittner               Czech Technical University in Prague
   vertex.h ----------------------------------------------------------------- */


#ifndef VERTEX_H
#define VERTEX_H


#include "../core/math.cuh"


int vertex_init(size_t, vec3_t *, size_t, vec3_t *);
int vertex_free(void);

int vertex_shader(void);

size_t vertex_get_size_vertices(void);
size_t vertex_get_size_normals(void);
vertex_t *vertex_get_vertices(void);
vec3_t *vertex_get_normals(void);
vec3_t *vertex_get_light_position(void);
mat4_t *vertex_get_view(void);
mat4_t *vertex_get_view_model(void);
mat4_t *vertex_get_view_model_inverse_transpose(void);
mat4_t *vertex_get_perspective_view_model(void);

#ifdef _DEBUG
void vertex_verbose(long long);
#endif /* _DEBUG */


/* -------------------------------------------------------------------------- */


#endif /* VERTEX_H */


/* vertex.h */