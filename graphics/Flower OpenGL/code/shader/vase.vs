#version 330

//-----------------------------------------------------------------------------
/**
* \file   vase.vs
* \author Jakub Profota
* \brief  Vase vertex shader.
*
* This file contains vase vertex shader.
*/
//-----------------------------------------------------------------------------


layout (location = 0) in vec3 in_position; // Input vertex position.
layout (location = 1) in vec3 in_normal;   // Input vertex normal.
layout (location = 2) in vec2 in_texcoord; // Input vertex texture coordinate.


smooth out vec3 v_position;  // Output vertex position.
smooth out vec3 v_normal;    // Output vertex normal.
smooth out vec2 v_texcoord;  // Output vertex texture coordinate.
out float v_camera_distance; // Output vertex distance from camera.


uniform mat4 P; // Perspective projection matrix.
uniform mat4 V; // View transformation matrix.
uniform mat4 M; // Model transformation matrix.

uniform sampler2D texture_normal; // Normal texture sampler.


/// Entry point of vertex shader.
void main() {
    gl_Position = P * V * M * vec4(in_position, 1.0);
    v_position = (M * vec4(in_position, 1.0)).xyz;
    v_normal = normalize((M * vec4(in_normal, 0.0)).xyz);
    v_texcoord = in_texcoord;
    v_camera_distance = length(vec3(V * M * vec4(in_position, 1.0)));
}


//-----------------------------------------------------------------------------
