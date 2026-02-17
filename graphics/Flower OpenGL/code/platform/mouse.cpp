//-----------------------------------------------------------------------------
/**
* \file   mouse.cpp
* \author Jakub Profota
* \brief  Mouse input system.
*
* This file contains mouse input related functions and callbacks.
*/
//-----------------------------------------------------------------------------


#include "pgr.h"

#include "window.h"

#include "../global.h"
#include "keyboard.h"


/// Handle the mouse press or mouse release event.
void mouse_callback(int button, int state, int mouse_x, int mouse_y) {
    if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) {
        unsigned char index;
        glReadPixels(mouse_x, scene->get_camera()->resolution.y - mouse_y - 1,
            1, 1, GL_STENCIL_INDEX, GL_UNSIGNED_BYTE, &index);

        if (index != 0) {
            if (keyboard_map[KEYBOARD_LEFT_SHIFT]) {
                Node *node = scene->select(index, 0);
            } else if (keyboard_map[KEYBOARD_LEFT_CONTROL]) {
                Node *node = scene->select(index, 1);
            } else {
                Node *node = scene->select(index, 2);
            }
        }
    }
}


/// Handle the mouse dragging event.
void mouse_motion_callback(int mouse_x, int mouse_y) {
}


/// Handle the mouse motion event.
void mouse_passive_motion_callback(int mouse_x, int mouse_y) {
    if (!scene->get_camera()->is_fixed && scene->get_camera()->is_active) {
        // Latitude.
        if (mouse_y != scene->get_camera()->resolution.y / 2) {
            float delta = 0.1f *
                (float)(mouse_y - scene->get_camera()->resolution.y / 2);
            scene->get_camera()->config.y += delta;
            if (scene->get_camera()->config.y < 0.5f) {
                scene->get_camera()->config.y = 0.5f;
            } else if (scene->get_camera()->config.y > 179.5f) {
                scene->get_camera()->config.y = 179.5f;
            }
        }

        // Longitude.
        if (mouse_x != scene->get_camera()->resolution.x / 2) {
            float delta = 0.1f *
                (float)(mouse_x - scene->get_camera()->resolution.x / 2);
            scene->get_camera()->config.x += delta;
            if (scene->get_camera()->config.x < 0.0f) {
                scene->get_camera()->config.x += 360.0f;
            } else if (scene->get_camera()->config.x >= 360.0f) {
                scene->get_camera()->config.x -= 360.0f;
            }
        }

        glutWarpPointer(scene->get_camera()->resolution.x / 2,
            scene->get_camera()->resolution.y / 2);

        scene->update();
    }
}


//-----------------------------------------------------------------------------
