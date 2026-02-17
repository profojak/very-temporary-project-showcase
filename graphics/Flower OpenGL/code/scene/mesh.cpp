//-----------------------------------------------------------------------------
/**
* \file   mesh.cpp
* \author Jakub Profota
* \brief  Mesh.
*
* This file contains mesh class.
*/
//-----------------------------------------------------------------------------


#include "mesh.h"

#include "../global.h"

#include <fstream>
#include <sstream>
#include <unordered_map>


/// Constructor.
Mesh::Mesh(std::string name,
    std::vector<GLfloat> &triangles, std::vector<GLuint> &indices,
    GLuint texture_color, GLuint texture_normal):
    name(name),
    vao(0), vbo(0), eao(0),
    texture_color(texture_color), texture_normal(texture_normal)
{
    n_vertices = triangles.size() / 8;
    n_indices = indices.size();

    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);
    glGenBuffers(1, &eao);

    glBindVertexArray(vao);

    // Copy vertices to the device.
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, triangles.size() * sizeof(GLfloat),
        triangles.data(), GL_STATIC_DRAW);
    CHECK_GL_ERROR();

    // Set shader attributes.
    GLsizei stride = 8 * sizeof(GLfloat);

    glEnableVertexAttribArray(ATTRIB_POSITION);
    glVertexAttribPointer(ATTRIB_POSITION, 3, GL_FLOAT, GL_FALSE,
        stride, (void *)(0 * sizeof(GLfloat)));
    CHECK_GL_ERROR();

    glEnableVertexAttribArray(ATTRIB_NORMAL);
    glVertexAttribPointer(ATTRIB_NORMAL, 3, GL_FLOAT, GL_FALSE,
        stride, (void *)(3 * sizeof(GLfloat)));
    CHECK_GL_ERROR();

    glEnableVertexAttribArray(ATTRIB_TEXCOORD);
    glVertexAttribPointer(ATTRIB_TEXCOORD, 2, GL_FLOAT, GL_FALSE,
        stride, (void *)(6 * sizeof(GLfloat)));
    CHECK_GL_ERROR();

    // Copy indices to the device.
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, eao);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(GLuint),
        indices.data(), GL_STATIC_DRAW);
    CHECK_GL_ERROR();
    n_triangles = indices.size() / 3;

    glBindVertexArray(0);
}


/// Destructor.
Mesh::~Mesh(void) {
    glDisableVertexAttribArray(ATTRIB_POSITION);
    glDisableVertexAttribArray(ATTRIB_NORMAL);
    glDisableVertexAttribArray(ATTRIB_TEXCOORD);
    glDeleteBuffers(1, &eao);
    glDeleteBuffers(1, &vbo);
    glDeleteVertexArrays(1, &vao);
}


/// Outputs the mesh information to standard output.
void Mesh::print(void) {
    std::cout << name << ":" << std::endl <<
        "    " << n_triangles << " triangles: " << n_vertices <<
        " vertices, " << n_indices << " indices" << std::endl <<
        "    textures: #" << texture_color << " color, #" << texture_normal <<
        " normal" << std::endl << "    #" << vao << " vao, #" << vbo <<
        " vbo, #" << eao << " eao" << std::endl;
}


//-----------------------------------------------------------------------------


/// Loads mesh from files.
Mesh *load_mesh(std::string file_name,
    std::string texture_color_name, std::string texture_normal_name) {
    std::ifstream file(file_name);

    // Check if mesh file exists.
    if (!file) {
        LOG_ERROR("Failed to open mesh file!");
        return nullptr;
    }

    std::unordered_map<unsigned int, GLuint> unique{};

    std::vector<GLfloat> vertices;
    std::vector<GLfloat> normals;
    std::vector<GLfloat> texcoords;
    std::vector<GLfloat> triangles;
    std::vector<GLuint> indices;

    std::string name;
    std::string line;

    // Iterate through each line.
    while (std::getline(file, line)) {
        // Mesh name.
        if (line.at(0) == 'o') {
            std::istringstream iss(&line.at(2));
            iss >> name;

        // Vertex.
        } else if (line.at(0) == 'v' && line.at(1) == ' ') {
            GLfloat x, y, z;

            // Check if vertex line has correct format.
            if (sscanf(line.data(), "v %f %f %f", &x, &y, &z) != 3) {
                LOG_ERROR("Failed to parse vertex line!");
                return nullptr;
            }

            vertices.push_back(x);
            vertices.push_back(y);
            vertices.push_back(z);

        // Normal.
        } else if (line.at(0) == 'v' && line.at(1) == 'n') {
            GLfloat x, y, z;

            // Check if normal line has correct format.
            if (sscanf(line.data(), "vn %f %f %f", &x, &y, &z) != 3) {
                LOG_ERROR("Failed to parse normal line!");
                return nullptr;
            }

            normals.push_back(x);
            normals.push_back(y);
            normals.push_back(z);

        // Texture coordinate.
        } else if (line.at(0) == 'v' && line.at(1) == 't') {
            GLfloat u, v;

            // Check if texture coordinate line has correct format.
            if (sscanf(line.data(), "vt %f %f", &u, &v) != 2) {
                LOG_ERROR("Failed to parse texture coordinate line!");
                return nullptr;
            }

            texcoords.push_back(u);
            texcoords.push_back(v);

        // Merge vertex, normal, and texture coordinate.
        } else if (line.at(0) == 'f') {
            GLuint v0, vt0, vn0;
            GLuint v1, vt1, vn1;
            GLuint v2, vt2, vn2;

            // Check if face line has correct format.
            if (sscanf(line.data(), "f %u/%u/%u %u/%u/%u %u/%u/%u",
                &v0, &vt0, &vn0, &v1, &vt1, &vn1, &v2, &vt2, &vn2) != 9) {
                LOG_ERROR("Failed to parse face line!");
                return nullptr;
            }

            // First vertex.
            float p0 = vertices.at(3 * (v0 - 1) + 0);
            float p1 = vertices.at(3 * (v0 - 1) + 1);
            float p2 = vertices.at(3 * (v0 - 1) + 2);
            float n0 = normals.at(3 * (vn0 - 1) + 0);
            float n1 = normals.at(3 * (vn0 - 1) + 1);
            float n2 = normals.at(3 * (vn0 - 1) + 2);
            float t0 = texcoords.at(2 * (vt0 - 1) + 0);
            float t1 = texcoords.at(2 * (vt0 - 1) + 1);
            unsigned int seed = hash(p0, p1, p2, n0, n1, n2, t0, t1);

            if (unique.count(seed) == 0) {
                unique[seed] = (GLuint)(triangles.size() / 8);
                triangles.push_back(p0);
                triangles.push_back(p1);
                triangles.push_back(p2);
                triangles.push_back(n0);
                triangles.push_back(n1);
                triangles.push_back(n2);
                triangles.push_back(t0);
                triangles.push_back(t1);
            }
            indices.push_back(unique[seed]);

            // Second vertex.
            p0 = vertices.at(3 * (v1 - 1) + 0);
            p1 = vertices.at(3 * (v1 - 1) + 1);
            p2 = vertices.at(3 * (v1 - 1) + 2);
            n0 = normals.at(3 * (vn1 - 1) + 0);
            n1 = normals.at(3 * (vn1 - 1) + 1);
            n2 = normals.at(3 * (vn1 - 1) + 2);
            t0 = texcoords.at(2 * (vt1 - 1) + 0);
            t1 = texcoords.at(2 * (vt1 - 1) + 1);
            seed = hash(p0, p1, p2, n0, n1, n2, t0, t1);

            if (unique.count(seed) == 0) {
                unique[seed] = (GLuint)(triangles.size() / 8);
                triangles.push_back(p0);
                triangles.push_back(p1);
                triangles.push_back(p2);
                triangles.push_back(n0);
                triangles.push_back(n1);
                triangles.push_back(n2);
                triangles.push_back(t0);
                triangles.push_back(t1);
            }
            indices.push_back(unique[seed]);

            // Third vertex.
            p0 = vertices.at(3 * (v2 - 1) + 0);
            p1 = vertices.at(3 * (v2 - 1) + 1);
            p2 = vertices.at(3 * (v2 - 1) + 2);
            n0 = normals.at(3 * (vn2 - 1) + 0);
            n1 = normals.at(3 * (vn2 - 1) + 1);
            n2 = normals.at(3 * (vn2 - 1) + 2);
            t0 = texcoords.at(2 * (vt2 - 1) + 0);
            t1 = texcoords.at(2 * (vt2 - 1) + 1);
            seed = hash(p0, p1, p2, n0, n1, n2, t0, t1);

            if (unique.count(seed) == 0) {
                unique[seed] = (GLuint)(triangles.size() / 8);
                triangles.push_back(p0);
                triangles.push_back(p1);
                triangles.push_back(p2);
                triangles.push_back(n0);
                triangles.push_back(n1);
                triangles.push_back(n2);
                triangles.push_back(t0);
                triangles.push_back(t1);
            }
            indices.push_back(unique[seed]);

        } else {
            LOG_ERROR("Failed to parse unknown line!");
            return nullptr;
        }
    }

    file.close();

    GLuint texture_color = 0;
    GLuint texture_normal = 0;

    // Textures.
    if (!texture_color_name.empty()) {
        texture_color = pgr::createTexture(texture_color_name);
    }

    if (!texture_normal_name.empty()) {
        texture_normal = pgr::createTexture(texture_normal_name);
    }

    // Return new mesh.
    return new Mesh(name, triangles, indices, texture_color, texture_normal);
}


/// Loads mesh from arrays.
Mesh *load_mesh(std::string name, const GLfloat in_vertices[],
    const GLuint in_indices[], GLuint n_indices,
    std::string texture_color_name, std::string texture_normal_name) {
    std::unordered_map<unsigned int, GLuint> unique{};

    std::vector<GLfloat> triangles;
    std::vector<GLuint> indices;

    for (int i = 0; i < n_indices; i++) {
        GLuint index = in_indices[i];

        float p0 = in_vertices[8 * index + 0];
        float p1 = in_vertices[8 * index + 1];
        float p2 = in_vertices[8 * index + 2];
        float n0 = in_vertices[8 * index + 3];
        float n1 = in_vertices[8 * index + 4];
        float n2 = in_vertices[8 * index + 5];
        float t0 = in_vertices[8 * index + 6];
        float t1 = in_vertices[8 * index + 7];
        float seed = hash(p0, p1, p2, n0, n1, n2, t0, t1);

        if (unique.count(seed) == 0) {
            unique[seed] = (GLuint)(triangles.size() / 8);
            triangles.push_back(p0);
            triangles.push_back(p1);
            triangles.push_back(p2);
            triangles.push_back(n0);
            triangles.push_back(n1);
            triangles.push_back(n2);
            triangles.push_back(t0);
            triangles.push_back(t1);
        }
        indices.push_back(unique[seed]);
    }

    GLuint texture_color = 0;
    GLuint texture_normal = 0;

    // Textures.
    if (!texture_color_name.empty()) {
        texture_color = pgr::createTexture(texture_color_name);
    }

    if (!texture_normal_name.empty()) {
        texture_normal = pgr::createTexture(texture_normal_name);
    }

    // Return new mesh.
    return new Mesh(name, triangles, indices, texture_color, texture_normal);
}


/// Hashes 8 float values: vertex position, normal, and texture coordinate.
unsigned int hash(float v0, float v1, float v2,
    float vn0, float vn1, float vn2, float vt0, float vt1) {
    unsigned int seed = 0;
    seed ^= std::hash<float>{}(v0) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    seed ^= std::hash<float>{}(v1) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    seed ^= std::hash<float>{}(v2) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    seed ^= std::hash<float>{}(vn0) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    seed ^= std::hash<float>{}(vn1) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    seed ^= std::hash<float>{}(vn2) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    seed ^= std::hash<float>{}(vt0) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    seed ^= std::hash<float>{}(vt1) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    return seed;
}


//-----------------------------------------------------------------------------
