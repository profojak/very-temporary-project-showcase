/* CUDA C Software Rendering Pipeline ---------------------------------------
   Bachelor Thesis                                              Jakub Profota
   Summer Semester 2022/3                    OI - Computer Games and Graphics
   Supervisor Jiri Bittner               Czech Technical University in Prague
   mutex.h ------------------------------------------------------------------ */


#ifndef MUTEX_H
#define MUTEX_H


typedef struct mutex_t {
    void *handle;
} mutex_t;


int mutex_create(mutex_t *);
int mutex_destroy(mutex_t *);

int mutex_lock(mutex_t *);
int mutex_unlock(mutex_t *);


/* -------------------------------------------------------------------------- */


#endif /* MUTEX_H */


/* mutex.h */