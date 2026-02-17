//-----------------------------------------------------------------------------
/**
* \file   window.cpp
* \author Jakub Profota
* \brief  Window system.
*
* This file contains window related functions and callbacks.
*/
//-----------------------------------------------------------------------------


#include "pgr.h"

#include "window.h"

#include "../global.h"


/// Draw the window contents.
void display_callback(void) {
    // Clear framebuffer.
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

    scene->draw();

    // Draw framebuffer.
    glutSwapBuffers();
}


/// Window was reshaped.
void reshape_callback(int width, int height) {
    // Set viewport size.
    glViewport(0, 0, width, height);

    // Set camera aspect ratio.
    scene->get_camera()->resolution.x = width;
    scene->get_camera()->resolution.y = height;
}


/// Periodically update scene.
void timer_callback(int value) {
    // Elapsed time in milliseconds.
    float elapsed_time = static_cast<float>(glutGet(GLUT_ELAPSED_TIME));

    // Register next timer callback.
    glutTimerFunc(33, timer_callback, 0);

    // Register display callback.
    glutPostRedisplay();
}


/// Clear the application data before closing.
void close_callback(void) {
}


/// Handle menu.
void menu_callback(int item) {
    switch (item) {
    case MENU_CAMERA_TOGGLE: {
        scene->get_camera()->is_fixed = !scene->get_camera()->is_fixed;
        glutWarpPointer(scene->get_camera()->resolution.x / 2,
            scene->get_camera()->resolution.y / 2);
        scene->get_camera()->view_center.x = scene->get_table()->size.x * 0.5f;
        scene->get_camera()->view_center.y = 0.0f;
        scene->get_camera()->view_center.z = scene->get_table()->size.y * 0.5f;
    } break;
    case MENU_REFLECTOR_TOGGLE: {
        scene->is_spotlight = !scene->is_spotlight;
    } break;
    case MENU_CAMERA_TOP: {
        if (scene->get_camera()->is_fixed) {
            break;
        }
        scene->get_camera()->is_active = false;
        scene->get_camera()->frame_counter = 60;
        scene->get_camera()->delta_position =
            glm::vec3(scene->get_table()->size.x * 0.5f, 1.5f, -0.5f) -
            scene->get_camera()->position;
        scene->get_camera()->delta_position /= 60.0f;
        scene->get_camera()->delta_view_center =
            glm::vec3(scene->get_table()->size.x * 0.5f, 0.2f,
                scene->get_table()->size.y * 0.5f);
        scene->get_camera()->config = glm::vec3(90.0f, 150.0f, 1.0f);
    } break;
    case MENU_CAMERA_FRONT: {
        if (scene->get_camera()->is_fixed) {
            break;
        }
        scene->get_camera()->is_active = false;
        scene->get_camera()->frame_counter = 60;
        scene->get_camera()->delta_position =
            glm::vec3(scene->get_table()->size.x * 0.5f - 0.1f, 0.15f,
                scene->get_table()->size.y + 0.7f) -
            scene->get_camera()->position;
        scene->get_camera()->delta_position /= 60.0f;
        scene->get_camera()->delta_view_center =
            glm::vec3(scene->get_table()->size.x * 0.5f, 0.2f,
                scene->get_table()->size.y * 0.5f);
        scene->get_camera()->config = glm::vec3(275.0f, 89.0f, 1.0f);
    } break;
    }
}


//-----------------------------------------------------------------------------
