#version 330

//-----------------------------------------------------------------------------
/**
* \file   skybox.vs
* \author Jakub Profota
* \brief  Skybox vertex shader.
*
* This file contains skybox vertex shader.
*/
//-----------------------------------------------------------------------------


layout (location = 0) in vec3 in_position; // Input vertex position.


smooth out vec3 v_texcoord; // Output vertex texture coordinate.


uniform mat4 P; // Perspective projection matrix.
uniform mat4 V; // View transformation matrix.


/// Entry point of vertex shader.
void main() {
    gl_Position = P * V * vec4(in_position, 1.0);
    v_texcoord = in_position;
}


//-----------------------------------------------------------------------------
