/* CUDA C Software Rendering Pipeline ---------------------------------------
   Bachelor Thesis                                              Jakub Profota
   Summer Semester 2022/3                    OI - Computer Games and Graphics
   Supervisor Jiri Bittner               Czech Technical University in Prague
   memory.c ----------------------------------------------------------------- */


#include "memory.h"

#include "win32.h"
#include <Windows.h>

#include <stdlib.h> // EXIT_SUCCESS, EXIT_FAILURE


/* Memory ------------------------------------------------------------------- */


/* Allocate memory */
void *mem_alloc(
    size_t size // Number of bytes to allocate
) {
    return (void *)HeapAlloc(
        GetProcessHeap(), // Handle to heap to allocate to
        HEAP_ZERO_MEMORY, // Zero out allocated memory
        size              // Number of bytes to allocate
    );
} /* mem_alloc */


/* Reallocate memory */
void *mem_realloc(
    void *src,  // Source to reallocate
    size_t size // Number of bytes to allocate
) {
    return (void *)HeapReAlloc(
        GetProcessHeap(), // Handle to heap to reallocate from
        0,                // No options
        src,              // Source to reallocate
        size              // Number of bytes to reallocate
    );
} /* mem_realloc */


/* Copy memory */
void *mem_copy(
    void *dst,       // Destination to copy to
    const void *src, // Source to copy from
    size_t size      // Number of bytes to copy
) {
    return memcpy(dst, src, size);
} /* mem_copy */


/* Set memory */
void *mem_set(
    void *dst,  // Destination to set
    int val,    // Value to set
    size_t size // Number of bytes to set
) {
    return memset(dst, val, size);
} /* mem_set */


/* Free memory */
int mem_free(
    void *dst // Destination to free
) {
    if (HeapFree(
        GetProcessHeap(), // Handle to heap to free from
        0,                // No options
        dst               // Destination to free
    ) != TRUE) {
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
} /* mem_free */


/* -------------------------------------------------------------------------- */


/* memory.c */