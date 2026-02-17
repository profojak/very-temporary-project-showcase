/* CUDA C Software Rendering Pipeline ---------------------------------------
   Bachelor Thesis                                              Jakub Profota
   Summer Semester 2022/3                    OI - Computer Games and Graphics
   Supervisor Jiri Bittner               Czech Technical University in Prague
   memory.h ----------------------------------------------------------------- */


#ifndef MEMORY_H
#define MEMORY_H


void *mem_alloc(unsigned long long);
void *mem_realloc(void *, unsigned long long);
void *mem_copy(void *, const void *, unsigned long long);
void *mem_set(void *, int, unsigned long long);
int mem_free(void *);


/* -------------------------------------------------------------------------- */


#endif /* MEMORY_H */


/* memory.h */