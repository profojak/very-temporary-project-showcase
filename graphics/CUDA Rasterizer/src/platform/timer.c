/* CUDA C Software Rendering Pipeline ---------------------------------------
   Bachelor Thesis                                              Jakub Profota
   Summer Semester 2022/3                    OI - Computer Games and Graphics
   Supervisor Jiri Bittner               Czech Technical University in Prague
   timer.c ------------------------------------------------------------------ */


#include "timer.h"

#include "print.h"

#include "win32.h"
#include <Windows.h>
#include <timeapi.h> // timeBeginPeriod, timeEndPeriod

#include <stdlib.h>  // EXIT_SUCCESS, EXIT_FAILURE


static LARGE_INTEGER ticks_frequency; // Frequency of performance counter


/* Init and free ------------------------------------------------------------ */


/* Initialize timer system */
extern int timer_init(
    void
) {
    // Request timer resolution of 1 millisecond to allow 1 millisecond Sleep
    // Use Sleep for n-1 milliseconds, skip fraction of last millisecond
    timeBeginPeriod(1);

    if (QueryPerformanceFrequency(&ticks_frequency) == 0) {
        #ifdef _DEBUG
        LOG_ERROR("Failed to retreive frequency of performance counter!");
        #endif /* _DEBUG */
        return EXIT_FAILURE;
    }

    #ifdef _DEBUG
    LOG_TEXT("Initialized timer system");
    LOG_TRACE;
    #endif /* _DEBUG */

    return EXIT_SUCCESS;
} /* timer_init */


/* Free timer system */
extern int timer_free(
    void
) {
    timeEndPeriod(1);

    #ifdef _DEBUG
    LOG_TEXT("Freed timer system");
    LOG_TRACE;
    #endif /* _DEBUG */

    return EXIT_SUCCESS;
} /* timer_free */


/* Update ------------------------------------------------------------------- */


/* Update timer */
extern int timer_update(
    timer_t *timer // Timer
) {
    LARGE_INTEGER ticks_now;
    if (QueryPerformanceCounter(&ticks_now) == 0) {
        #ifdef _DEBUG
        LOG_ERROR("Failed to retreive current value of performance counter!");
        #endif /* _DEBUG */
        return EXIT_FAILURE;
    }
    timer->time_elapsed = ticks_now.QuadPart - timer->ticks_last;
    timer->ticks_last = ticks_now.QuadPart;

    // Divide by ticks frequency after conversion to microseconds
    // Guard against loss of precision
    timer->time_elapsed *= 1000000;
    timer->time_elapsed /= ticks_frequency.QuadPart;

    return EXIT_SUCCESS;
} /* timer_update */


/* -------------------------------------------------------------------------- */


/* timer.c */