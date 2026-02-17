//-----------------------------------------------------------------------------
/**
* \file   main.cpp
* \author Jakub Profota
* \brief  Entry point of the application.
*
* This file contains the application entry point. Program execution begins and
* ends here.
*/
//-----------------------------------------------------------------------------


#include "pgr.h"

#include "global.h"
#include "platform/keyboard.h"
#include "platform/mouse.h"
#include "platform/window.h"
#include "scene/library.h"
#include "scene/mesh.h"
#include "scene/scene.h"


/// Map of pressed keys.
char keyboard_map[128] = { 0 };
/// Shader library.
ShaderLibrary *shader_library = nullptr;
/// Mesh library.
MeshLibrary *mesh_library = nullptr;
/// Scene.
Scene *scene = nullptr;


/// Entry point of the application.
/**
* \param[in] argc Number of command line arguments.
* \param[in] argv Array of argument strings.
* \return         Zero on success.
*/
int main(int argc, char** argv) {
    // Initialize the GLUT windowing system.
    glutInit(&argc, argv);
    glutInitContextVersion(pgr::OGL_VER_MAJOR, pgr::OGL_VER_MINOR);
    glutInitContextFlags(GLUT_FORWARD_COMPATIBLE);
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);

    // Create window.
    glutInitWindowSize(1366, 768);
    glutCreateWindow("PGR: profojak");

    // Set callbacks.
    glutDisplayFunc(display_callback);
    glutReshapeFunc(reshape_callback);
    glutCloseFunc(close_callback);

    glutKeyboardFunc(keyboard_press_callback);
    glutKeyboardUpFunc(keyboard_release_callback);
    glutSpecialFunc(special_press_callback);
    glutSpecialUpFunc(special_release_callback);
    glutMouseFunc(mouse_callback);
    glutMotionFunc(mouse_motion_callback);
    glutPassiveMotionFunc(mouse_passive_motion_callback);

    // Initialize menu.
    glutCreateMenu(menu_callback);
    glutAddMenuEntry("Toggle Camera Freemode", MENU_CAMERA_TOGGLE);
    glutAddMenuEntry("Toggle Camera Reflector", MENU_REFLECTOR_TOGGLE);
    glutAddMenuEntry("Camera Top", MENU_CAMERA_TOP);
    glutAddMenuEntry("Camera Front", MENU_CAMERA_FRONT);
    glutAttachMenu(GLUT_RIGHT_BUTTON);

    // Initialize PGR framework.
    if (!pgr::initialize(pgr::OGL_VER_MAJOR, pgr::OGL_VER_MINOR)) {
        pgr::dieWithError("PGR initialization failed!");
    }

    // Initialize OpenGL.
    glClearColor(0.5f, 0.5f, 0.5f, 1.0f);
    glEnable(GL_DEPTH_TEST);

    // Initialize shader and mesh library.
    shader_library = load_shader_library();
    mesh_library = load_mesh_library();

    // Initialize scene.
    scene = load_scene("scene/save.txt");
    scene->update();

    shader_library->print();
    mesh_library->print();
    scene->print();

    // Main loop.
    glutTimerFunc(33, timer_callback, 0);
    glutMainLoop();

    return EXIT_SUCCESS;
}


//-----------------------------------------------------------------------------
