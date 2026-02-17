/* CUDA C Software Rendering Pipeline ---------------------------------------
   Bachelor Thesis                                              Jakub Profota
   Summer Semester 2022/3                    OI - Computer Games and Graphics
   Supervisor Jiri Bittner               Czech Technical University in Prague
   mutex.c ------------------------------------------------------------------ */


#include "mutex.h"

#include "print.h"

#include "win32.h"
#include <Windows.h>

#include <stdlib.h> // EXIT_SUCCESS, EXIT_FAILURE


/* Create and destroy ------------------------------------------------------- */


/* Create mutex */
extern int mutex_create(
    mutex_t *mutex // Mutex
) {
    mutex->handle = (void *)CreateMutexA(
        0, // Handle cannot be inherited by child processes
        0, // Calling thread does not obtain ownership of mutex
        0  // Mutex object created without name
    );

    if (mutex->handle == 0) {
        #ifdef _DEBUG
        LOG_ERROR("Failed to create mutex!");
        #endif /* _DEBUG */
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
} /* mutex_create */


/* Destroy mutex */
extern int mutex_destroy(
    mutex_t *mutex // Mutex
) {
    if (CloseHandle((HANDLE)mutex->handle) != TRUE) {
        #ifdef _DEBUG
        LOG_ERROR("Failed to destroy mutex!");
        #endif /* _DEBUG */
        return EXIT_FAILURE;
    }
    mutex->handle = 0;

    return EXIT_SUCCESS;
} /* mutex_destroy */


/* Lock and unlock ---------------------------------------------------------- */


/* Mutex lock */
extern int mutex_lock(
    mutex_t *mutex // Mutex
) {
    if (WaitForSingleObject((HANDLE)mutex->handle, INFINITE) == WAIT_FAILED) {
        #ifdef _DEBUG
        LOG_ERROR("Failed to lock mutex!");
        #endif /* _DEBUG */
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
} /* mutex_lock */


/* Mutex unlock */
extern int mutex_unlock(
    mutex_t *mutex // Mutex
) {
    if (ReleaseMutex((HANDLE)mutex->handle) != TRUE) {
        #ifdef _DEBUG
        LOG_ERROR("Failed to unlock mutex!");
        #endif /* _DEBUG */
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
} /* mutex_unlock */


/* -------------------------------------------------------------------------- */


/* mutex.c */