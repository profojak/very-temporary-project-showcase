/* CUDA C Software Rendering Pipeline ---------------------------------------
   Bachelor Thesis                                              Jakub Profota
   Summer Semester 2022/3                    OI - Computer Games and Graphics
   Supervisor Jiri Bittner               Czech Technical University in Prague
   print.h ------------------------------------------------------------------ */


#ifndef PRINT_H
#define PRINT_H


enum print_e {
    // Macros
    PRINT_ERROR,    // Error
    PRINT_WARNING,  // Warning
    PRINT_INFO,     // Info
    PRINT_TEXT,     // Text
    PRINT_DEBUG,    // Debug
    PRINT_TRACE,    // Trace

    // Foreground
    PRINT_WHITE_FG, // White
    PRINT_GRAY_FG,  // Gray
    PRINT_BLUE_FG,  // Blue
    PRINT_AQUA_FG,  // Aqua
    PRINT_CYAN_FG,  // Cyan
    PRINT_GREEN_FG, // Green
    PRINT_LIME_FG,  // Lime

    // Background
    PRINT_WHITE_BG, // White
    PRINT_GRAY_BG,  // Gray
    PRINT_BLUE_BG,  // Blue
    PRINT_AQUA_BG,  // Aqua
    PRINT_CYAN_BG,  // Cyan
    PRINT_GREEN_BG, // Green
    PRINT_LIME_BG   // Lime
};


int print_init(void);
int print_free(void);

int print(enum print_e, const char *, ...);

char *filename(const char *);


/* Print -------------------------------------------------------------------- */


#ifdef _DEBUG


#define LOG_DEBUG(msg, ...) {                                                  \
    print(PRINT_DEBUG, (const char *)" " msg, ##__VA_ARGS__);                  \
}


#define LOG_EMPTY {                                                            \
    print(PRINT_DEBUG, "");                                                    \
}


#define LOG_TRACE {                                                            \
    print(PRINT_TRACE, "     %s:%s:%i",                                        \
        __FUNCTION__, filename(__FILE__), __LINE__);                           \
}


#define LOG_ERROR(msg, ...) {                                                  \
    print(PRINT_ERROR, (const char *)" " msg, ##__VA_ARGS__);                  \
    LOG_TRACE;                                                                 \
}


#define LOG_WARNING(msg, ...) {                                                \
    print(PRINT_WARNING, (const char *)" " msg, ##__VA_ARGS__);                \
}


#define LOG_INFO(msg, ...) {                                                   \
    print(PRINT_INFO, (const char *)" " msg, ##__VA_ARGS__);                   \
}


#define LOG_TEXT(msg, ...) {                                                   \
    print(PRINT_TEXT, (const char *)" " msg, ##__VA_ARGS__);                   \
}


/* Assert ------------------------------------------------------------------- */


#define ASSERT(expr) {                                                         \
    if (expr) {                                                                \
    } else {                                                                   \
        LOG_ERROR("%s", #expr);                                                \
    }                                                                          \
}


#define ASSERT_MSG(expr, msg, ...) {                                           \
    if (expr) {                                                                \
    } else {                                                                   \
        LOG_ERROR("%s " msg, #expr, ##__VA_ARGS__);                            \
    }                                                                          \
}


#endif /* _DEBUG */


/* -------------------------------------------------------------------------- */


#endif /* PRINT_H */


/* print.h */