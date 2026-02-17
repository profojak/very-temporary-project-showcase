#version 330

//-----------------------------------------------------------------------------
/**
* \file   skybox.fs
* \author Jakub Profota
* \brief  Skybox fragment shader.
*
* This file contains skybox fragment shader.
*/
//-----------------------------------------------------------------------------


smooth in vec3 v_texcoord; // Input fragment texture coordinate.


out vec4 f_color; // Output fragment color.


uniform samplerCube skybox; // Skybox sampler.


/// Entry point of fragment shader.
void main() {
    f_color = 0.6 * vec4(0.5, 0.5, 0.5, 1.0) + 0.4 * texture(skybox, v_texcoord);
}


//-----------------------------------------------------------------------------
