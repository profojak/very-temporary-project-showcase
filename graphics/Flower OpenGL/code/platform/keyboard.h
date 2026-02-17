//-----------------------------------------------------------------------------
/**
* \file   keyboard.h
* \author Jakub Profota
* \brief  Keyboard input system.
*
* This file contains keyboard input related functions and callbacks.
*/
//-----------------------------------------------------------------------------


#ifndef KEYBOARD_H
#define KEYBOARD_H


/// Enum for application key codes.
enum KeyboardEnum {
    KEYBOARD_LEFT_SHIFT = 0,
    KEYBOARD_LEFT_CONTROL = 1,
    KEYBOARD_RIGHT_CONTROL = 2,
    KEYBOARD_RIGHT_SHIFT = 3,
    KEYBOARD_LEFT_ARROW = 4,
    KEYBOARD_RIGHT_ARROW = 5,
    KEYBOARD_UP_ARROW = 6,
    KEYBOARD_DOWN_ARROW = 7
};


/// Handle the key press event.
/**
* This function is called whenever a key on the keyboard is pressed.
*
* \param[in] key     ASCII code of the pressed key.
* \param[in] mouse_x Mouse cursor X position.
* \param[in] mouse_y Mouse cursor Y position.
*/
void keyboard_press_callback(unsigned char key, int mouse_x, int mouse_y);


/// Handle the key release event.
/**
* This function is called whenever a key on the keyboard is released.
*
* \param[in] key     ASCII code of the released key.
* \param[in] mouse_x Mouse cursor X position.
* \param[in] mouse_y Mouse cursor Y position.
*/
void keyboard_release_callback(unsigned char key, int mouse_x, int mouse_y);


/// Handle the special key press event.
/**
* This function is called whenever a special key on the keyboard is pressed.
*
* \param[in] key     GLUT constant code of the pressed key.
* \param[in] mouse_x Mouse cursor X position.
* \param[in] mouse_y Mouse cursor Y position.
*/
void special_press_callback(int key, int mouse_x, int mouse_y);


/// Handle the special key release event.
/**
* This function is called whenever a special key on the keyboard is released.
*
* \param[in] key     GLUT constant code of the released key.
* \param[in] mouse_x Mouse cursor X position.
* \param[in] mouse_y Mouse cursor Y position.
*/
void special_release_callback(int key, int mouse_x, int mouse_y);


//-----------------------------------------------------------------------------


#endif // !KEYBOARD_H
