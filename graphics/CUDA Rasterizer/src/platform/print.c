/* CUDA C Software Rendering Pipeline ---------------------------------------
   Bachelor Thesis                                              Jakub Profota
   Summer Semester 2022/3                    OI - Computer Games and Graphics
   Supervisor Jiri Bittner               Czech Technical University in Prague
   print.c ------------------------------------------------------------------ */


#include "print.h"

#include "../core/define.h"
#include "memory.h"

#include "win32.h"
#include <Windows.h>
#include <Shlwapi.h> // PathFindFileNameA

#include <stdio.h>  // vsnprintf
#include <stdlib.h> // EXIT_SUCCESS, EXIT_FAILURE


static unsigned short level_attrib[20] = {
    BACKGROUND_RED,                                            // Error
    BACKGROUND_RED | BACKGROUND_GREEN,                         // Warning
    BACKGROUND_RED | BACKGROUND_GREEN | BACKGROUND_BLUE,       // Info
    BACKGROUND_INTENSITY,                                      // Text
    FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE,       // Debug
    FOREGROUND_INTENSITY,                                      // Trace

    FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE,       // White
    FOREGROUND_INTENSITY,                                      // Gray
    FOREGROUND_BLUE | FOREGROUND_INTENSITY,                    // Blue
    FOREGROUND_BLUE | FOREGROUND_GREEN,                        // Aqua
    FOREGROUND_BLUE | FOREGROUND_GREEN | FOREGROUND_INTENSITY, // Cyan
    FOREGROUND_GREEN,                                          // Green
    FOREGROUND_GREEN | FOREGROUND_INTENSITY,                   // Lime

    BACKGROUND_RED | BACKGROUND_GREEN | BACKGROUND_BLUE,       // White
    BACKGROUND_INTENSITY,                                      // Gray
    BACKGROUND_BLUE | BACKGROUND_INTENSITY,                    // Blue
    BACKGROUND_BLUE | BACKGROUND_GREEN,                        // Aqua
    BACKGROUND_BLUE | BACKGROUND_GREEN | BACKGROUND_INTENSITY, // Cyan
    BACKGROUND_GREEN,                                          // Green
    BACKGROUND_GREEN | BACKGROUND_INTENSITY                    // Lime
};


static int console(enum print_e, const char *);


/* Init and free ------------------------------------------------------------ */


/* Initialize print system */
extern int print_init(
    void
) {
    if (AllocConsole() != TRUE) {
        return EXIT_FAILURE;
    }

    #ifdef _DEBUG
    LOG_WARNING("CUDA C Software Rendering Pipeline");
    LOG_EMPTY;
    LOG_EMPTY;
    LOG_INFO("Main thread initialization");
    LOG_TEXT("Initialized print system");
    LOG_TRACE;
    #endif /* _DEBUG */

    return EXIT_SUCCESS;
} /* print_init */


/* Free print system */
extern int print_free(
    void
) {
    #ifdef _DEBUG
    LOG_TEXT("Freed print system");
    LOG_TRACE;
    LOG_EMPTY;
    LOG_INFO("End of console output");
    LOG_EMPTY;
    #endif /* _DEBUG */

    return EXIT_SUCCESS;
} /* print_free */


/* Print -------------------------------------------------------------------- */


/* Print message */
extern int print(
    enum print_e print_level, // Error level of message
    const char *msg,          // Message with format
    ...                       // Arguments with data to print
) {
    char output[PRINT_MAX_LEN + 1];
    if (print_level < PRINT_WHITE_FG) {
        mem_set((void *)output, ' ', sizeof output);
    } else {
        mem_set((void *)output, 0, sizeof output);
    }

    va_list args;
    va_start(args, msg);
    char buffer[PRINT_MAX_LEN];
    int n_written = vsnprintf(buffer, PRINT_MAX_LEN, msg, args);
    mem_copy(output, buffer, n_written);
    va_end(args);

    if (print_level < PRINT_TEXT) {
        output[PRINT_MAX_LEN - 1] = '\n';
        output[PRINT_MAX_LEN] = 0;
    } else if (print_level < PRINT_WHITE_FG) {
        size_t len = strlen(buffer);
        output[len + 1] = '\n';
        output[len + 2] = 0;
    }

    console(print_level, output);

    return EXIT_SUCCESS;
} /* print */


/* Write message to console */
static int console(
    enum print_e print_level, // Error level of message
    const char *src           // Source string to write
) {
    HANDLE handle_console = GetStdHandle(STD_OUTPUT_HANDLE);
    SetConsoleTextAttribute(handle_console, level_attrib[print_level]);

    size_t length = strlen((const char *)src);
    DWORD n_written = 0;
    if (WriteConsoleA(
        handle_console, // Handle to console
        src,            // String to write to console
        (DWORD)length,  // Number of characters to write
        &n_written,     // Receive number of characters written
        0               // Always 0
    ) != TRUE) {
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
} /* print_output */


/* File --------------------------------------------------------------------- */


/* Return file name from path to file */
extern char *filename(
    const char *path // Source string with path
) {
    return PathFindFileNameA(path);
} /* file_name */


/* -------------------------------------------------------------------------- */


/* print.c */