//-----------------------------------------------------------------------------
/**
* \file   library.h
* \author Jakub Profota
* \brief  Libraries.
*
* This file contains mesh and shader libraries.
*/
//-----------------------------------------------------------------------------


#ifndef LIBRARY_H
#define LIBRARY_H


#include "mesh.h"
#include "program.h"

#include <map>
#include <vector>


/// Holds all indexes to shaders vector.
enum ShaderEnum {
    SHADER_FLOWER,
    SHADER_VASE,
    SHADER_SAUCER,
    SHADER_TABLE,
    SHADER_SKYBOX,
    SHADER_BUTTERFLY,
    SHADER_BUG,
    SHADER_SPIDER,
    SHADER_COUNT
};


/// Holds all indexes to meshes vector.
enum MeshEnum {
    FLOWER_BERGENIA,
    FLOWER_CROCUS,
    FLOWER_GERANIUM,
    FLOWER_HIBISCOUS,
    FLOWER_LILIES,
    FLOWER_ROSE,
    FLOWER_SUNFLOWER,
    VASE_BRONZE,
    VASE_EGYPT,
    VASE_JAPAN,
    VASE_PORCELAIN,
    VASE_TERRACOTA,
    SAUCER,
    BUTTERFLY,
    BUG,
    SPIDER,
    TABLE,
    MESH_COUNT
};


enum ColorEnum {
    COLOR_WHITE,
    COLOR_GRAY,
    COLOR_BLACK,
    COLOR_RED,
    COLOR_GREEN,
    COLOR_BLUE,
    COLOR_YELLOW,
    COLOR_CYAN,
    COLOR_MAGENTA,
    COLOR_COUNT
};


//-----------------------------------------------------------------------------


class ShaderLibrary {
public:
    /// Constructor.
    ShaderLibrary(void);


    /// Destructor.
    ~ShaderLibrary(void);


    /// Outputs the shader library to standard output.
    void print(void);


    /// Sets shader to specified index.
    /**
    * \param[in] index Index to set shader to.
    * \param[in] mesh  Shader.
    */
    void set_shader(enum ShaderEnum index, Shader* shader) {
        this->shaders[index] = shader;
    }


    /// Gets shader specified by the name.
    /**
    * \param[in] name Name of the shader.
    */
    Shader *get_shader(std::string name) {
        return this->shaders[string_to_enum[name]];
    }


private:
    std::map<std::string, ShaderEnum> string_to_enum; ///< String to enum map.
    std::vector<Shader *> shaders;                    ///< Vector of shaders.
};


class MeshLibrary {
public:
    /// Constructor.
    MeshLibrary(void);


    /// Destructor.
    ~MeshLibrary(void);


    /// Outputs the mesh library to standard output.
    void print(void);


    /// Sets mesh to specified index.
    /**
    * \param[in] index Index to set mesh to.
    * \param[in] mesh  Mesh.
    */
    void set_mesh(enum MeshEnum index, Mesh* mesh) {
        this->meshes[index] = mesh;
    }


    /// Gets mesh specified by the name.
    /**
    * \param[in] name Name of the mesh.
    */
    Mesh *get_mesh(std::string name) {
        return this->meshes[string_to_enum[name]];
    }


    /// Gets color specified by the name.
    /**
    * \param[in] name Name of the color.
    */
    glm::vec3 get_color(std::string name) {
        return this->colors[string_to_color[name]];
    }


private:
    std::map<std::string, MeshEnum> string_to_enum;   ///< String to enum map.
    std::map<std::string, ColorEnum> string_to_color; ///< String to color map.
    std::vector<Mesh *> meshes;                       ///< Vector of meshes.
    std::vector<glm::vec3> colors;                    ///< Vector of colors.
};


//-----------------------------------------------------------------------------


/// Loads all shaders.
/**
* This function parses all shader files used in the program and stores them in
* ShaderLibrary.
*/
ShaderLibrary *load_shader_library(void);


/// Loads all meshes.
/**
* This function parses all mesh files used in the program and stores them in
* MeshLibrary.
*/
MeshLibrary *load_mesh_library(void);


//-----------------------------------------------------------------------------


#endif // !LIBRARY_H
