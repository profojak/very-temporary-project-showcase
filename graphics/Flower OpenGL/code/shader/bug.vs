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

    float f_elapsed = float(elapsed);
    vec2 offset = vec2(0.0, sin(f_elapsed * 0.01) / 2.75 + 0.5);
    v_texcoord = in_texcoord + offset;
    v_camera_distance = length(vec3(V * M * vec4(in_position, 1.0)));
}


//-----------------------------------------------------------------------------
