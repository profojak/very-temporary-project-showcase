//-----------------------------------------------------------------------------
/**
* \file   keyboard.cpp
* \author Jakub Profota
* \brief  Keyboard input system.
*
* This file contains keyboard input related functions and callbacks.
*/
//-----------------------------------------------------------------------------


#include "keyboard.h"

#include "../global.h"

#include "pgr.h"


/// Enum for ASCII key codes.
enum AsciiEnum {
    ASCII_ESC = 27,
    ASCII_LEFT_SHIFT = 112,
    ASCII_LEFT_CONTROL = 114,
    ASCII_RIGHT_CONTROL = 115,
    ASCII_RIGHT_SHIFT = 113,
    ASCII_LEFT_ARROW = 100,
    ASCII_RIGHT_ARROW = 102,
    ASCII_UP_ARROW = 101,
    ASCII_DOWN_ARROW = 103
};


//-----------------------------------------------------------------------------


/// Handle the key press event.
void keyboard_press_callback(unsigned char key, int mouse_x, int mouse_y) {
    if (key == ASCII_ESC) {
        glutLeaveMainLoop();
    } else if (key >= 32 && key <= 127) {
        keyboard_map[key] = 1;
    }

    scene->update();
}


/// Handle the key release event.
void keyboard_release_callback(unsigned char key, int mouse_x, int mouse_y) {
    if (key >= 32 && key <= 127) {
        keyboard_map[key] = 0;
    }
}


/// Handle the special key press event.
void special_press_callback(int key, int mouse_x, int mouse_y) {
    if (key == ASCII_LEFT_SHIFT) {
        keyboard_map[KEYBOARD_LEFT_SHIFT] = 1;
    } else if (key == ASCII_LEFT_CONTROL) {
        keyboard_map[KEYBOARD_LEFT_CONTROL] = 1;
    } else if (key == ASCII_RIGHT_CONTROL) {
        keyboard_map[KEYBOARD_RIGHT_CONTROL] = 1;
    } else if (key == ASCII_RIGHT_SHIFT) {
        keyboard_map[KEYBOARD_RIGHT_SHIFT] = 1;
    } else if (key == ASCII_LEFT_ARROW) {
        keyboard_map[KEYBOARD_LEFT_ARROW] = 1;
    } else if (key == ASCII_RIGHT_ARROW) {
        keyboard_map[KEYBOARD_RIGHT_ARROW] = 1;
    } else if (key == ASCII_UP_ARROW) {
        keyboard_map[KEYBOARD_UP_ARROW] = 1;
    } else if (key == ASCII_DOWN_ARROW) {
        keyboard_map[KEYBOARD_DOWN_ARROW] = 1;
    }

    scene->update();
}


/// Handle the special key release event.
void special_release_callback(int key, int mouse_x, int mouse_y) {
    if (key == ASCII_LEFT_SHIFT) {
        keyboard_map[KEYBOARD_LEFT_SHIFT] = 0;
    } else if (key == ASCII_LEFT_CONTROL) {
        keyboard_map[KEYBOARD_LEFT_CONTROL] = 0;
    } else if (key == ASCII_RIGHT_CONTROL) {
        keyboard_map[KEYBOARD_RIGHT_CONTROL] = 0;
    } else if (key == ASCII_RIGHT_SHIFT) {
        keyboard_map[KEYBOARD_RIGHT_SHIFT] = 0;
    } else if (key == ASCII_LEFT_ARROW) {
        keyboard_map[KEYBOARD_LEFT_ARROW] = 0;
    } else if (key == ASCII_RIGHT_ARROW) {
        keyboard_map[KEYBOARD_RIGHT_ARROW] = 0;
    } else if (key == ASCII_UP_ARROW) {
        keyboard_map[KEYBOARD_UP_ARROW] = 0;
    } else if (key == ASCII_DOWN_ARROW) {
        keyboard_map[KEYBOARD_DOWN_ARROW] = 0;
    }
}


//-----------------------------------------------------------------------------
