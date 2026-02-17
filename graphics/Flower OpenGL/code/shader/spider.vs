#version 330

//-----------------------------------------------------------------------------
/**
* \file   bug.vs
* \author Jakub Profota
* \brief  Bug vertex shader.
*
* This file contains bug vertex shader.
*/
//-----------------------------------------------------------------------------


layout (location = 0) in vec3 in_position; // Input vertex position.
layout (location = 2) in vec2 in_texcoord; // Input vertex texture coordinate.


smooth out vec3 v_position;  // Output vertex position.
smooth out vec2 v_texcoord;  // Output vertex texture coordinate.
out float v_camera_distance; // Output vertex distance from camera.


uniform mat4 P;      // Perspective projection matrix.
uniform mat4 V;      // View transformation matrix.
uniform mat4 M;      // Model transformation matrix.
uniform int elapsed; // Number of elapsed frames.


/// Entry point of vertex shader.
void main() {
    gl_Position = P * V * M * vec4(in_position, 1.0);
    v_texcoord = in_texcoord;
    v_texcoord.x /= 8.0;
    v_texcoord += vec2(((elapsed % 64) / 8) / 8.0, 0.0);
    v_camera_distance = length(vec3(V * M * vec4(in_position, 1.0)));
}


//-----------------------------------------------------------------------------
