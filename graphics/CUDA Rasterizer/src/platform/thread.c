/* CUDA C Software Rendering Pipeline ---------------------------------------
   Bachelor Thesis                                              Jakub Profota
   Summer Semester 2022/3                    OI - Computer Games and Graphics
   Supervisor Jiri Bittner               Czech Technical University in Prague
   thread.c ----------------------------------------------------------------- */


#include "thread.h"

#include "print.h"

#include "win32.h"
#include <Windows.h>

#include <stdlib.h> // EXIT_SUCCESS, EXIT_FAILURE


/* Create and destroy ------------------------------------------------------- */


/* Create thread */
extern int thread_create(
    thread_t *thread, // Thread
    void *function,   // Entry function of thread
    void *args        // Arguments to pass to thread
) {
    thread->handle = (void *)CreateThread(
        0,                   // Handle cannot be inherited by child processes
        0,                   // Use default stack size
        (LPTHREAD_START_ROUTINE)function, // Function to execute
        args,                // Variable to pass to thread
        0,                   // Flags that control creation of thread
        (DWORD *)&thread->id // Variable that receives thread identifier
    );
    if (thread->handle == 0) {
        #ifdef _DEBUG
        LOG_ERROR("Failed to create thread!");
        #endif /* _DEBUG */
        return EXIT_FAILURE;
    }

    #ifdef _DEBUG
    LOG_TEXT("Created thread");
    LOG_TRACE;
    #endif /* _DEBUG */

    return EXIT_SUCCESS;
} /* thread_create */


/* Destroy thread */
extern int thread_destroy(
    thread_t *thread // Thread
) {
    if (WaitForSingleObject((HANDLE)thread->handle, INFINITE) == WAIT_FAILED) {
        #ifdef _DEBUG
        LOG_ERROR("Failed to destroy thread!");
        #endif /* _DEBUG */
        return EXIT_FAILURE;
    }
    thread->handle = 0;

    #ifdef _DEBUG
    LOG_TEXT("Destroyed thread");
    LOG_TRACE;
    #endif /* _DEBUG */

    return EXIT_SUCCESS;
} /* thread_destroy */


/* Sleep -------------------------------------------------------------------- */


/* Sleep on calling thread */
extern inline void thread_sleep(
    unsigned long milliseconds // Milliseconds to sleep for
) {
    Sleep(milliseconds);
} /* thread_sleep */


/* -------------------------------------------------------------------------- */


/* thread.c */