#version 330

//-----------------------------------------------------------------------------
/**
* \file   bug.fs
* \author Jakub Profota
* \brief  Bug fragment shader.
*
* This file contains bug fragment shader.
*/
//-----------------------------------------------------------------------------


smooth in vec3 v_position;  // Input fragment position.
smooth in vec2 v_texcoord;  // Input fragment texture coordinate.
in float v_camera_distance; // Input fragment distance from camera.


out vec4 f_color; // Output fragment color.


uniform sampler2D texture_color;  // Color texture sampler.


/// Entry point of fragment shader.
void main() {
    float fog = clamp(exp2(-0.25 * v_camera_distance), 0.0, 1.0);
    f_color = (1 - fog) * vec4(0.5, 0.5, 0.5, 0.0) +
        fog * texture(texture_color, v_texcoord);
}


//-----------------------------------------------------------------------------
