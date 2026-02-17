//-----------------------------------------------------------------------------
/**
* \file   mesh.h
* \author Jakub Profota
* \brief  Mesh.
*
* This file contains mesh class.
*/
//-----------------------------------------------------------------------------


#ifndef MESH_H
#define MESH_H

#include "pgr.h"


//-----------------------------------------------------------------------------


/// Class that holds mesh data.
class Mesh {
public:
    /// Constructor.
    /**
    * \param[in] index          Mesh name.
    * \param[in] triangles      Vector of triangles.
    * \param[in] indices        Vector of indices.
    * \param[in] texture_color  Color texture.
    * \param[in] texture_normal Normal texture.
    */
    Mesh(std::string name,
        std::vector<GLfloat> &triangles, std::vector<GLuint> &indices,
        GLuint texture_color, GLuint texture_normal);


    /// Destructor.
    ~Mesh(void);


    /// Outputs the mesh to standard output.
    void print(void);


    /// Gets number of triangles.
    GLuint get_n_triangles(void) {
        return n_triangles;
    }


    /// Gets vertex array object.
    GLuint get_vao(void) {
        return vao;
    }


    /// Gets color texture.
    GLuint get_texture_color(void) {
        return texture_color;
    }


    /// Gets normal texture.
    GLuint get_texture_normal(void) {
        return texture_normal;
    }


protected:
    std::string name; ///< Mesh name.

    GLuint n_triangles; ///< Number of triangles.
    GLuint n_vertices;  ///< Number of unique vertices.
    GLuint n_indices;   ///< Number of unique indices.

    GLuint texture_color;  ///< Color texture.
    GLuint texture_normal; ///< Normal texture.

    GLuint vao; ///< Vertex array object.
    GLuint vbo; ///< Vertex buffer object.
    GLuint eao; ///< Element array buffer object.
};


//-----------------------------------------------------------------------------


/// Loads mesh from files.
/**
* This function opens files, parses the contents and loads the mesh. Files must
* have either a valid Wavefront .obj format, or image format. Only vertices,
* normals, and texture coordinates are supported by this loader. The face
* polygon must be triangle. Image support is dictated by the PGR framework.
* 
* \param[in] file_name           Mesh file name.
* \param[in] texture_color_name  Color texture file name.
* \param[in] texture_normal_name Normal texture file name.
*/
Mesh *load_mesh(std::string file_name,
    std::string texture_color_name, std::string texture_normal_name);


/// Loads mesh from arrays.
/**
* This function loads the mesh from hardcoded arrays. The face polygon must be
* triangle.
* 
* \param[in] name     Name of the mesh.
* \param[in] vertices Array of vertices.
* \param[in] vertices Array of indices.
* \param[in] vertices Size of index array.
*/
Mesh *load_mesh(std::string name, const GLfloat vertices[],
    const GLuint indices[], GLuint n_indices,
    std::string texture_color_name, std::string texture_normal_name);


/// Hashes 8 float values: vertex position, normal, and texture coordinate.
/**
* \param[in] v0  Vertex position x.
* \param[in] v1  Vertex position y.
* \param[in] v2  Vertex position z.
* \param[in] vn0 Vertex normal x.
* \param[in] vn1 Vertex normal y.
* \param[in] vn2 Vertex normal z.
* \param[in] vt0 Vertex texture coordinate u.
* \param[in] vt1 Vertex texture coordinate v.
*/
unsigned int hash(float v0, float v1, float v2,
    float vn0, float vn1, float vn2, float vt0, float vt1);


//-----------------------------------------------------------------------------


#endif // !MESH_H
