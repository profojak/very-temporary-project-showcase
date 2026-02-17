/* CUDA C Software Rendering Pipeline ---------------------------------------
   Bachelor Thesis                                              Jakub Profota
   Summer Semester 2022/3                    OI - Computer Games and Graphics
   Supervisor Jiri Bittner               Czech Technical University in Prague
   rop.h -------------------------------------------------------------------- */


#ifndef ROP_H
#define ROP_H


int rop_init(void);
int rop_free(void);

int raster_operation(void);

unsigned int *rop_get_framebuffer(void);

#ifdef _DEBUG
void rop_verbose(long long);
#endif /* _DEBUG */


/* -------------------------------------------------------------------------- */


#endif /* ROP_H */


/* rop.h */