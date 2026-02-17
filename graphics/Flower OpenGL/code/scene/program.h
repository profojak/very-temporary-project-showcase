//-----------------------------------------------------------------------------
/**
* \file   program.h
* \author Jakub Profota
* \brief  Shader program.
*
* This file contains shader program class.
*/
//-----------------------------------------------------------------------------


#ifndef PROGRAM_H
#define PROGRAM_H


#include "pgr.h"


/// Holds all indexes to shader attributes.
enum ShaderAttribEnum {
    ATTRIB_POSITION,
    ATTRIB_NORMAL,
    ATTRIB_TEXCOORD
};


/// Holds all indexes to shader uniforms.
enum ShaderUniformEnum {
    UNIFORM_P,
    UNIFORM_V,
    UNIFORM_M,
    UNIFORM_ELAPSED,
    UNIFORM_COLOR,
    TEXTURE_COLOR,
    TEXTURE_NORMAL,
    SKYBOX,

    MATERIAL_AMBIENT,
    MATERIAL_DIFFUSE,
    MATERIAL_SPECULAR,
    MATERIAL_SHININESS,

    DIRECTIONAL_AMBIENT,
    DIRECTIONAL_DIFFUSE,
    DIRECTIONAL_SPECULAR,
    DIRECTIONAL_DIRECTION,

    SPOTLIGHT_ON,
    SPOTLIGHT_AMBIENT,
    SPOTLIGHT_DIFFUSE,
    SPOTLIGHT_SPECULAR,
    SPOTLIGHT_POSITION,
    SPOTLIGHT_DIRECTION,
    SPOTLIGHT_CONE,
    SPOTLIGHT_EXPONENT,

    POINT_AMBIENT,
    POINT_DIFFUSE,
    POINT_SPECULAR,
    POINT_POSITION,
    POINT_EXPONENT
};


//-----------------------------------------------------------------------------


/// Class that holds shader configuration.
class Shader {
public:
    /// Constructor.
    /**
    * \param[in] program_id Shader program id.
    */
    Shader(std::string name, GLuint program_id);


    /// Destructor.
    ~Shader(void);


    /// Outputs the shader to standard output.
    void print(void);


    /// Gets program id.
    GLuint get_id(void) {
        return program_id;
    }


    /// Gets uniform location.
    /**
    * \param[in] uniform Uniform index.
    */
    GLint get_uniform(enum ShaderUniformEnum uniform);


private:
    std::string name;  ///< Shader name.
    GLuint program_id; ///< Shader program id.

    GLint uniform_P; ///< Perspective projection matrix uniform location.
    GLint uniform_V; ///< View transformation matrix uniform location.
    GLint uniform_M; ///< Model transformation matrix uniform location.
    GLint uniform_elapsed; ///< Elapsed number of frames.

    GLint uniform_color; ///< Color.

    GLint texture_color;  ///< Color texture sampler uniform location.
    GLint texture_normal; ///< Normal texture sampler uniform location.
    GLint skybox;         ///< Skybox cubemap texture.

    GLint material_ambient;   // Ambient material component.
    GLint material_diffuse;   // Diffuse material component.
    GLint material_specular;  // Specular material component.
    GLint material_shininess; // Material shininess.

    GLint directional_ambient;   // Ambient light component.
    GLint directional_diffuse;   // Diffuse light component.
    GLint directional_specular;  // Specular light component.
    GLint directional_direction; // Light direction.

    GLint spotlight_on;        // Whether reflector spotlight light is on.
    GLint spotlight_ambient;   // Ambient light component.
    GLint spotlight_diffuse;   // Diffuse light component.
    GLint spotlight_specular;  // Specular light component.
    GLint spotlight_position;  // Light position.
    GLint spotlight_direction; // Light direction.
    GLint spotlight_cone;      // Cosine of the spotlight half angle.
    GLint spotlight_exponent;  // Distribution of the energy within the light cone.

    GLint point_ambient;   // Ambient light component.
    GLint point_diffuse;   // Diffuse light component.
    GLint point_specular;  // Specular light component.
    GLint point_position;  // Light position.
    GLint point_exponent;  // Distribution of the energy within the light cone.
};


//-----------------------------------------------------------------------------


/// Loads shader program from files.
/**
* \param[in] name Shader name.
*/
Shader *load_shader(std::string name);


//-----------------------------------------------------------------------------


#endif // !PROGRAM_H
