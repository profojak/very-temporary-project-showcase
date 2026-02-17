//-----------------------------------------------------------------------------
/**
* \file   window.h
* \author Jakub Profota
* \brief  Window system.
*
* This file contains window related functions and callbacks.
*/
//-----------------------------------------------------------------------------


#ifndef WINDOW_H
#define WINDOW_H


/// Enum for menu entries.
enum MenuEnum {
    MENU_CAMERA_TOGGLE,
    MENU_REFLECTOR_TOGGLE,
    MENU_CAMERA_TOP,
    MENU_CAMERA_FRONT
};


/// Draw the window contents.
/**
* This function draws the scene with objects to the window.
*/
void display_callback(void);


/// Update on window reshape.
/**
* This function updates the application to the new window resolution.
*
* \param[in] new_width  New window width.
* \param[in] new_height New window height.
*/
void reshape_callback(int new_width, int new_height);


/// Periodically update scene.
/**
* This function periodically updates the scene state and creates a display
* event.
*
* \param[in] value Value set when timer callback was registered.
*/
void timer_callback(int value);


/// Clear the application data before closing.
/**
* This function handles the application's request to close by cleaning
* everything.
*/
void close_callback(void);


/// Handle menu.
/*
* \param[in] item Selected item.
*/
void menu_callback(int item);


//-----------------------------------------------------------------------------


#endif // !WINDOW_H
