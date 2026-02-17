#version 330

//-----------------------------------------------------------------------------
/**
* \file   saucer.vs
* \author Jakub Profota
* \brief  Saucer vertex shader.
*
* This file contains saucer vertex shader.
*/
//-----------------------------------------------------------------------------


layout (location = 0) in vec3 in_position; // Input vertex position.


smooth out vec3 v_position;  // Output vertex position.
out float v_camera_distance; // Output vertex distance from camera.


uniform mat4 P; // Perspective projection matrix.
uniform mat4 V; // View transformation matrix.
uniform mat4 M; // Model transformation matrix.


/// Entry point of vertex shader.
void main() {
    gl_Position = P * V * M * vec4(in_position, 1.0);
    v_position = (M * vec4(in_position, 1.0)).xyz;
    v_camera_distance = length(vec3(V * M * vec4(in_position, 1.0)));
}


//-----------------------------------------------------------------------------
