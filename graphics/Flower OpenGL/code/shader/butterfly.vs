#version 330

//-----------------------------------------------------------------------------
/**
* \file   butterfly.vs
* \author Jakub Profota
* \brief  Butterfly vertex shader.
*
* This file contains butterfly vertex shader.
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
    vec4 position = P * V * M * vec4(in_position, 1.0);
    float f_elapsed = float(elapsed);
    position.y += sin(f_elapsed * 0.03) * 0.25;
    if (in_position.x != 0) {
        position.y += sin(f_elapsed) * 0.05;
    }
    gl_Position = position;

    v_position = (M * vec4(in_position, 1.0)).xyz;
    v_texcoord = in_texcoord;
    v_camera_distance = length(vec3(V * M * vec4(in_position, 1.0)));
}


//-----------------------------------------------------------------------------
