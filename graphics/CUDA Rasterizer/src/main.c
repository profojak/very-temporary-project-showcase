/* CUDA C Software Rendering Pipeline ---------------------------------------
   Bachelor Thesis                                              Jakub Profota
   Summer Semester 2022/3                    OI - Computer Games and Graphics
   Supervisor Jiri Bittner               Czech Technical University in Prague
   main.c ------------------------------------------------------------------- */


#include "platform/win32.h"
#include <Windows.h>

#include <stdlib.h> // system


// Entry point of software rendering pipeline: pipeline.cu
extern int pipeline(void *, void *, int);


/* Entry -------------------------------------------------------------------- */


/* Entry point of program */
extern int WINAPI WinMain(
    HINSTANCE h_instance,      // Handle to identify executable
    HINSTANCE h_prev_instance, // Always 0
    PSTR p_cmd_line,           // String of command line arguments
    int n_cmd_show             // Window flag to show or hide window
) {
    // Bypass unreferenced parameter warning
    h_prev_instance = h_prev_instance;

    int ret = pipeline((void *)h_instance, (void *)p_cmd_line, n_cmd_show);

    // Wait for key press
    system("pause");
    return ret;
} /* WinMain */


/* -------------------------------------------------------------------------- */


/* main.c */