/* CUDA C Software Rendering Pipeline ---------------------------------------
   Bachelor Thesis                                              Jakub Profota
   Summer Semester 2022/3                    OI - Computer Games and Graphics
   Supervisor Jiri Bittner               Czech Technical University in Prague
   thread.h ----------------------------------------------------------------- */


#ifndef THREAD_H
#define THREAD_H


typedef struct thread_t {
    void *handle;          // Handle to thread
    unsigned long long id; // Unique thread identification
} thread_t;


int thread_create(thread_t *, void *, void *);
int thread_destroy(thread_t *);

void thread_sleep(unsigned long);


/* -------------------------------------------------------------------------- */


#endif /* THREAD_H */


/* thread.h */