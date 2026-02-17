//-----------------------------------------------------------------------------
/**
* \file   global.h
* \author Jakub Profota
* \brief  Global data.
*
* This file holds global data used throughout the application.
*/
//-----------------------------------------------------------------------------


#ifndef GLOBAL_H
#define GLOBAL_H

#include "scene/library.h"
#include "scene/scene.h"

#include <iostream>


/// Map of pressed keys.
extern char keyboard_map[128];
/// Shader library.
extern ShaderLibrary *shader_library;
/// Mesh library.
extern MeshLibrary *mesh_library;
/// Scene.
extern Scene *scene;


//-----------------------------------------------------------------------------


/// Error output.
#define LOG_ERROR(string) {                                                   \
    std::cout << "[\033[31mERROR\033[0m] " << __FUNCTION__ << ": " <<         \
        string << std::endl;                                                  \
}


/// Info output.
#define LOG_INFO(string) {                                                    \
    std::cout << "[\033[32mINFO\033[0m] " << __FUNCTION__ << ": " <<          \
        string << std::endl;                                                  \
}


//-----------------------------------------------------------------------------


#endif // !GLOBAL_H
