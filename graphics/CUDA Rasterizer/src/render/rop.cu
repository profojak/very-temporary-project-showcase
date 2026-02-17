/* CUDA C Software Rendering Pipeline ---------------------------------------
   Bachelor Thesis                                              Jakub Profota
   Summer Semester 2022/3                    OI - Computer Games and Graphics
   Supervisor Jiri Bittner               Czech Technical University in Prague
   rop.cu ------------------------------------------------------------------- */


#include "rop.h"

#include "../core/define.h"
#include "../platform/print.h"
#include "../platform/window.h"


static unsigned int *host_framebuffer = { 0 };
static unsigned int *device_framebuffer = { 0 };


/* Init and free ------------------------------------------------------------ */


/* Initialize raster operation */
extern int rop_init(
    void
) {
    host_framebuffer = *window_get_buffer();
    if (host_framebuffer == 0) {
        #ifdef _DEBUG
        LOG_ERROR("Failed to initialize host framebuffer!");
        #endif /* _DEBUG */
        return EXIT_FAILURE;
    }

    if (cudaMalloc(
        &device_framebuffer,
        WINDOW_WIDTH * WINDOW_HEIGHT * sizeof(unsigned int)
    ) != cudaSuccess) {
        #ifdef _DEBUG
        LOG_ERROR("Failed to initialize device framebuffer!");
        #endif /* _DEBUG */
        return EXIT_FAILURE;
    }

    #ifdef _DEBUG
    LOG_TEXT("Initialized raster operation");
    LOG_TRACE;
    #endif /* _DEBUG */

    return EXIT_SUCCESS;
} /* rop_init */


/* Free raster operation */
extern int rop_free(
    void
) {
    host_framebuffer = 0;
    if (cudaFree((void *)device_framebuffer) != cudaSuccess) {
        #ifdef _DEBUG
        LOG_ERROR("Failed to free device framebuffer!");
        #endif /* _DEBUG */
        return EXIT_FAILURE;
    }
    device_framebuffer = 0;

    #ifdef _DEBUG
    LOG_TEXT("Freed raster operation");
    LOG_TRACE;
    #endif /* _DEBUG */

    return EXIT_SUCCESS;
} /* rop_free */


/* Raster operation --------------------------------------------------------- */


/* Draw contents of framebuffer to window */
extern inline int raster_operation(
    void
) {
    if (cudaMemcpy(
        host_framebuffer,
        device_framebuffer,
        WINDOW_WIDTH * WINDOW_HEIGHT * sizeof(unsigned int),
        cudaMemcpyDeviceToHost
    ) != cudaSuccess) {
        #ifdef _DEBUG
        LOG_ERROR("Failed to copy framebuffer from device to host!");
        #endif /* _DEBUG */
        return EXIT_FAILURE;
    }

    window_callback_invalidate();

    return EXIT_SUCCESS;
} /* raster_operation */


/* Get ---------------------------------------------------------------------- */


/* Get device framebuffer */
extern inline unsigned int *rop_get_framebuffer(
    void
) {
    return device_framebuffer;
} /* rop_get_framebuffer */


/* Verbose ------------------------------------------------------------------ */


#ifdef _DEBUG


/* Verbose output */
extern void rop_verbose(
    long long time // Elapsed time
) {
    LOG_TEXT("Raster operation");
    LOG_TRACE;
    print(PRINT_LIME_BG, " %lld ", time);
    print(PRINT_GREEN_BG, " us \n");
} /* rop_verbose */


#endif /* _DEBUG */


/* -------------------------------------------------------------------------- */


/* rop.cu */