/* CUDA C Software Rendering Pipeline ---------------------------------------
   Bachelor Thesis                                              Jakub Profota
   Summer Semester 2022/3                    OI - Computer Games and Graphics
   Supervisor Jiri Bittner               Czech Technical University in Prague
   pixel.h ------------------------------------------------------------------ */


#ifndef PIXEL_H
#define PIXEL_H


#include "../core/math.cuh"


int pixel_init(unsigned int *, vec3_t *, rgb_t *, rgb_t *, rgb_t *);
int pixel_free(void);

int pixel_shader(void);

fragment_t *pixel_get_fragments(void);
int *pixel_get_light_shininess(void);

#ifdef _DEBUG
void pixel_verbose(long long);
#endif /* _DEBUG */


/* -------------------------------------------------------------------------- */


#endif /* PIXEL_H */


/* pixel.h */