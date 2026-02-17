//-----------------------------------------------------------------------------
/**
* \file   mouse.h
* \author Jakub Profota
* \brief  Mouse input system.
*
* This file contains mouse input related functions and callbacks.
*/
//-----------------------------------------------------------------------------


#ifndef MOUSE_H
#define MOUSE_H


/// Handle the mouse press or mouse release event.
/**
* This function is called whenever a mouse key is pressed or released.
*
* \param[in] button  GLUT_LEFT_BUTTON, GLUT_MIDDLE_BUTTON, GLUT_RIGHT_BUTTON.
* \param[in] state   GLUT_DOWN when pressed, GLUT_UP when released.
* \param[in] mouse_x Mouse cursor X position.
* \param[in] mouse_y Mouse cursor Y position.
*/
void mouse_callback(int button, int state, int mouse_x, int mouse_y);


/// Handle the mouse dragging event.
/**
* This function is called whenever a mouse motion is registered while any
* mouse button is pressed.
*
* \param[in] mouse_x Mouse cursor X position.
* \param[in] mouse_y Mouse cursor Y position.
*/
void mouse_motion_callback(int mouse_x, int mouse_y);


/// Handle the mouse motion event.
/**
* This function is called whenever a mouse motion is registered.
*
* \param[in] mouse_x Mouse cursor X position.
* \param[in] mouse_y Mouse cursor Y position.
*/
void mouse_passive_motion_callback(int mouse_x, int mouse_y);


//-----------------------------------------------------------------------------


#endif // !MOUSE_H
