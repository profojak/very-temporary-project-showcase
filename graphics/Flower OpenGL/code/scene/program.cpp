//-----------------------------------------------------------------------------
/**
* \file   program.cpp
* \author Jakub Profota
* \brief  Shader program.
*
* This file contains shader program class.
*/
//-----------------------------------------------------------------------------


#include "program.h"

#include "../global.h"


/// Constructor.
Shader::Shader(std::string name, GLuint program_id):
    name(name), program_id(program_id)
{
    // Attribute.
    glBindAttribLocation(program_id, ATTRIB_POSITION, "in_position");
    CHECK_GL_ERROR();
    glBindAttribLocation(program_id, ATTRIB_NORMAL, "in_normal");
    CHECK_GL_ERROR();
    glBindAttribLocation(program_id, ATTRIB_TEXCOORD, "in_texcoord");
    CHECK_GL_ERROR();

    // Uniform.
    uniform_P = glGetUniformLocation(program_id, "P");
    uniform_V = glGetUniformLocation(program_id, "V");
    uniform_M = glGetUniformLocation(program_id, "M");
    uniform_elapsed = glGetUniformLocation(program_id, "elapsed");

    uniform_color = glGetUniformLocation(program_id, "color");

    texture_color = glGetUniformLocation(program_id, "texture_color");
    texture_normal = glGetUniformLocation(program_id, "texture_normal");
    skybox = glGetUniformLocation(program_id, "skybox");

    material_ambient = glGetUniformLocation(program_id,
        "material.ambient");
    material_diffuse = glGetUniformLocation(program_id,
        "material.diffuse");
    material_specular = glGetUniformLocation(program_id,
        "material.specular");
    material_shininess = glGetUniformLocation(program_id,
        "material.shininess");

    directional_ambient = glGetUniformLocation(program_id,
        "directional.ambient");
    directional_diffuse = glGetUniformLocation(program_id,
        "directional.diffuse");
    directional_specular = glGetUniformLocation(program_id,
        "directional.specular");
    directional_direction = glGetUniformLocation(program_id,
        "directional.direction");

    spotlight_on = glGetUniformLocation(program_id,
        "spotlight.on");
    spotlight_ambient = glGetUniformLocation(program_id,
        "spotlight.ambient");
    spotlight_diffuse = glGetUniformLocation(program_id,
        "spotlight.diffuse");
    spotlight_specular = glGetUniformLocation(program_id,
        "spotlight.specular");
    spotlight_position = glGetUniformLocation(program_id,
        "spotlight.position");
    spotlight_direction = glGetUniformLocation(program_id,
        "spotlight.direction");
    spotlight_cone = glGetUniformLocation(program_id,
        "spotlight.cone");
    spotlight_exponent = glGetUniformLocation(program_id,
        "spotlight.exponent");

    point_ambient = glGetUniformLocation(program_id,
        "point.ambient");
    point_diffuse = glGetUniformLocation(program_id,
        "point.diffuse");
    point_specular = glGetUniformLocation(program_id,
        "point.specular");
    point_position = glGetUniformLocation(program_id,
        "point.position");
    point_exponent = glGetUniformLocation(program_id,
        "point.exponent");
}


/// Destructor.
Shader::~Shader(void) {
    pgr::deleteProgramAndShaders(program_id);
}


/// Outputs the shader to standard output.
void Shader::print(void) {
    std::cout << name << ":" << std::endl <<
        "    #" << program_id << " program id" << std::endl <<
        "    uniforms: #" << uniform_P << " P, #" << uniform_V << " V, #" <<
        uniform_M << " M, " << " camera pos, " <<
        uniform_color << " color" << std::endl << "    textures: #" <<
        texture_color << " color, #" << texture_normal << " normal, #" <<
        skybox << " skybox" << std::endl;
}


/// Gets uniform location.
GLint Shader::get_uniform(enum ShaderUniformEnum uniform) {
    switch (uniform) {
    case UNIFORM_P:
        return uniform_P;
    case UNIFORM_V:
        return uniform_V;
    case UNIFORM_M:
        return uniform_M;
    case UNIFORM_ELAPSED:
        return uniform_elapsed;
    case UNIFORM_COLOR:
        return uniform_color;
    case TEXTURE_COLOR:
        return texture_color;
    case TEXTURE_NORMAL:
        return texture_normal;
    case SKYBOX:
        return skybox;
    case MATERIAL_AMBIENT:
        return material_ambient;
    case MATERIAL_DIFFUSE:
        return material_diffuse;
    case MATERIAL_SPECULAR:
        return material_specular;
    case MATERIAL_SHININESS:
        return material_shininess;
    case DIRECTIONAL_AMBIENT:
        return directional_ambient;
    case DIRECTIONAL_DIFFUSE:
        return directional_diffuse;
    case DIRECTIONAL_SPECULAR:
        return directional_specular;
    case DIRECTIONAL_DIRECTION:
        return directional_direction;
    case SPOTLIGHT_ON:
        return spotlight_on;
    case SPOTLIGHT_AMBIENT:
        return spotlight_ambient;
    case SPOTLIGHT_DIFFUSE:
        return spotlight_diffuse;
    case SPOTLIGHT_SPECULAR:
        return spotlight_specular;
    case SPOTLIGHT_POSITION:
        return spotlight_position;
    case SPOTLIGHT_DIRECTION:
        return spotlight_direction;
    case SPOTLIGHT_CONE:
        return spotlight_cone;
    case SPOTLIGHT_EXPONENT:
        return spotlight_exponent;
    case POINT_AMBIENT:
        return point_ambient;
    case POINT_DIFFUSE:
        return point_diffuse;
    case POINT_SPECULAR:
        return point_specular;
    case POINT_POSITION:
        return point_position;
    case POINT_EXPONENT:
        return point_exponent;
    default:
        LOG_ERROR("Invalid uniform!");
        return -1;
    }
}


//-----------------------------------------------------------------------------


/// Loads shader program from files.
Shader *load_shader(std::string name) {
    GLuint shader_list[] = {
        pgr::createShaderFromFile(GL_VERTEX_SHADER,
        std::string("code/shader/").append(name).append(".vs")),
        pgr::createShaderFromFile(GL_FRAGMENT_SHADER,
        std::string("code/shader/").append(name).append(".fs")),
        0
    };

    // Return new shader.
    return new Shader(name, pgr::createProgram(shader_list));
}


//-----------------------------------------------------------------------------
