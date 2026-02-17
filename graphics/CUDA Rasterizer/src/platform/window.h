/* CUDA C Software Rendering Pipeline ---------------------------------------
   Bachelor Thesis                                              Jakub Profota
   Summer Semester 2022/3                    OI - Computer Games and Graphics
   Supervisor Jiri Bittner               Czech Technical University in Prague
   window.h ----------------------------------------------------------------- */


#ifndef WINDOW_H
#define WINDOW_H


enum status_e {
    WINDOW_NOT_INITIALIZED, // Not initialized
    WINDOW_ACTIVE,          // Active
    WINDOW_SHUTTING_DOWN    // Shutting down
};


int window_init(void *, int);
int window_free(void);

void window_callback_message(void);
void window_callback_invalidate(void);

void window_set_status(enum status_e);

enum status_e window_get_status(void);
unsigned int **window_get_buffer(void);

/* -------------------------------------------------------------------------- */


#endif /* WINDOW_H */


/* window.h */