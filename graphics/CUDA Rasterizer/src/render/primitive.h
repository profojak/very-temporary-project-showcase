/* CUDA C Software Rendering Pipeline ---------------------------------------
   Bachelor Thesis                                              Jakub Profota
   Summer Semester 2022/3                    OI - Computer Games and Graphics
   Supervisor Jiri Bittner               Czech Technical University in Prague
   primitive.h -------------------------------------------------------------- */


#ifndef PRIMITIVE_H
#define PRIMITIVE_H


#include "../core/math.cuh"


int primitive_init(vertex_t *, vec3_t *, size_t, ivec2_t *);
int primitive_free(void);

int primitive_assembly(void);

size_t primitive_get_size_triangles(void);
triangle_t *primitive_get_triangles(void);

#ifdef _DEBUG
void primitive_verbose(long long);
#endif /* _DEBUG */


/* -------------------------------------------------------------------------- */


#endif /* PRIMITIVE_H */


/* primitive.h */