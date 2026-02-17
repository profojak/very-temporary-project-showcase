/* CUDA C Software Rendering Pipeline ---------------------------------------
   Bachelor Thesis                                              Jakub Profota
   Summer Semester 2022/3                    OI - Computer Games and Graphics
   Supervisor Jiri Bittner               Czech Technical University in Prague
   rasterize.h -------------------------------------------------------------- */


#ifndef RASTERIZE_H
#define RASTERIZE_H


#include "../core/math.cuh"


int rasterize_init(size_t, triangle_t *, fragment_t *);
int rasterize_free(void);

int rasterization(void);

#ifdef _DEBUG
void rasterize_verbose(long long);
#endif /* _DEBUG */


/* -------------------------------------------------------------------------- */


#endif /* RASTERIZE_H */


/* rasterize.h */