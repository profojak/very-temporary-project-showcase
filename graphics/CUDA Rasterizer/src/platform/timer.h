/* CUDA C Software Rendering Pipeline ---------------------------------------
   Bachelor Thesis                                              Jakub Profota
   Summer Semester 2022/3                    OI - Computer Games and Graphics
   Supervisor Jiri Bittner               Czech Technical University in Prague
   timer.h ------------------------------------------------------------------ */


#ifndef TIMER_H
#define TIMER_H


typedef struct timer_t {
    long long ticks_last;   // Ticks on last update call
    long long time_elapsed; // Elapsed microseconds since last update call
} timer_t;


int timer_init(void);
int timer_free(void);

int timer_update(timer_t *);


/* -------------------------------------------------------------------------- */


#endif /* TIMER_H */


/* timer.h */