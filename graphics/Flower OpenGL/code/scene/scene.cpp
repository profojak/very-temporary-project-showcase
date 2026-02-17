//-----------------------------------------------------------------------------
/**
* \file   scene.cpp
* \author Jakub Profota
* \brief  Scene.
*
* This file contains scene hierarchy logic and scene loader. The root of the
* scene is the table node, which holds a number of saucers, butterflies and a
* single lamp. Each saucer can then hold a single vase node, which holds a
* number of flowers.
*/
//-----------------------------------------------------------------------------


#include "scene.h"

#include "../global.h"
#include "node.h"


#include <fstream>
#include <sstream>


/// Constructor.
Skybox::Skybox(void) {
    // Set textures.
    glGenTextures(1, &texture_cubemap);
    glBindTexture(GL_TEXTURE_CUBE_MAP, texture_cubemap);

    pgr::loadTexImage2D("scene/skybox/right.jpg", GL_TEXTURE_CUBE_MAP_POSITIVE_X);
    pgr::loadTexImage2D("scene/skybox/left.jpg", GL_TEXTURE_CUBE_MAP_NEGATIVE_X);    pgr::loadTexImage2D("scene/skybox/right.jpg", GL_TEXTURE_CUBE_MAP_POSITIVE_X);
    pgr::loadTexImage2D("scene/skybox/top.jpg", GL_TEXTURE_CUBE_MAP_POSITIVE_Y);
    pgr::loadTexImage2D("scene/skybox/bottom.jpg", GL_TEXTURE_CUBE_MAP_NEGATIVE_Y);
    pgr::loadTexImage2D("scene/skybox/front.jpg", GL_TEXTURE_CUBE_MAP_POSITIVE_Z);
    pgr::loadTexImage2D("scene/skybox/back.jpg", GL_TEXTURE_CUBE_MAP_NEGATIVE_Z);

    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);

    glBindTexture(GL_TEXTURE_CUBE_MAP, 0);

    // Set geometry.
    GLfloat vertices[] = {
        -1.0f,  1.0f, -1.0f,
        -1.0f, -1.0f, -1.0f,
        1.0f, -1.0f, -1.0f,
        1.0f, -1.0f, -1.0f,
        1.0f,  1.0f, -1.0f,
        -1.0f,  1.0f, -1.0f,

        -1.0f, -1.0f,  1.0f,
        -1.0f, -1.0f, -1.0f,
        -1.0f,  1.0f, -1.0f,
        -1.0f,  1.0f, -1.0f,
        -1.0f,  1.0f,  1.0f,
        -1.0f, -1.0f,  1.0f,

        1.0f, -1.0f, -1.0f,
        1.0f, -1.0f,  1.0f,
        1.0f,  1.0f,  1.0f,
        1.0f,  1.0f,  1.0f,
        1.0f,  1.0f, -1.0f,
        1.0f, -1.0f, -1.0f,

        -1.0f, -1.0f,  1.0f,
        -1.0f,  1.0f,  1.0f,
        1.0f,  1.0f,  1.0f,
        1.0f,  1.0f,  1.0f,
        1.0f, -1.0f,  1.0f,
        -1.0f, -1.0f,  1.0f,

        -1.0f,  1.0f, -1.0f,
        1.0f,  1.0f, -1.0f,
        1.0f,  1.0f,  1.0f,
        1.0f,  1.0f,  1.0f,
        -1.0f,  1.0f,  1.0f,
        -1.0f,  1.0f, -1.0f,

        -1.0f, -1.0f, -1.0f,
        -1.0f, -1.0f,  1.0f,
        1.0f, -1.0f, -1.0f,
        1.0f, -1.0f, -1.0f,
        -1.0f, -1.0f,  1.0f,
        1.0f, -1.0f,  1.0f
    };

    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);

    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, 108 * sizeof(GLfloat),
        vertices, GL_STATIC_DRAW);
    CHECK_GL_ERROR();

    glEnableVertexAttribArray(ATTRIB_POSITION);
    glVertexAttribPointer(ATTRIB_POSITION, 3, GL_FLOAT, GL_FALSE, 0, 0);
    CHECK_GL_ERROR();

    glBindVertexArray(0);
}


/// Destructor.
Skybox::~Skybox(void) {
}


//-----------------------------------------------------------------------------


/// Constructor.
Scene::Scene(void):
    table(nullptr)
{
    camera = new Camera();
    skybox = new Skybox();
    directional = new Light();
    spotlight = new Light();
    point = new Light();

    directional->ambient = glm::vec3(0.0f);
    directional->diffuse = glm::vec3(0.8f, 0.75f, 0.0f);
    directional->specular = glm::vec3(0.8f);
    directional->direction = glm::vec3(-1.0f, 2.0f, -1.0f);

    is_spotlight = true;
    spotlight->ambient = glm::vec3(0.0f);
    spotlight->diffuse = glm::vec3(0.75f, 1.0f, 1.0f);
    spotlight->specular = glm::vec3(1.0f, 1.0f, 1.0f);
    spotlight->position = camera->position;
    spotlight->direction = camera->view_center;
    spotlight->cone = 0.95f;
    spotlight->exponent = 0.0f;

    point->ambient = glm::vec3(0.0f);
    point->diffuse = glm::vec3(1.0f, 0.0f, 0.0f);
    point->specular = glm::vec3(1.0f, 0.0f, 0.0f);
    point->position = glm::vec3(0.0f, 0.0f, 0.0f);
    point->exponent = 0.5f;
}


/// Destructor.
Scene::~Scene(void) {
    delete(table);
    delete(camera);
}


/// Outputs the scene hierarchy to standard output.
void Scene::print(void) {
    LOG_INFO("");
    this->get_table()->print();
}


/// Sets table node.
/**
 * \param[in] table Set table node.
 */
void Scene::set_table(Table* table) {
    this->table = table;
    glm::vec2 size = this->table->size;
    float t = size.x / size.y;

    const GLfloat table_vertices[] = {
        0.0f,     0.0f,   0.0f,  0.0f,  1.0f,  0.0f, 0.0f, 0.0f, //  0
        size.x,   0.0f,   0.0f,  0.0f,  1.0f,  0.0f, 0.0f, 1.0f, //  1
        0.0f,     0.0f, size.y,  0.0f,  1.0f,  0.0f, 1.0f, 0.0f, //  2
        size.x,   0.0f, size.y,  0.0f,  1.0f,  0.0f, 1.0f, 1.0f, //  3
        0.0f,     0.0f,   0.0f,  0.0f,  0.0f, -1.0f, 0.0f, 0.0f, //  4
        size.x,   0.0f,   0.0f,  0.0f,  0.0f, -1.0f, 0.0f, 0.0f, //  5
        0.0f,   -0.03f,   0.0f,  0.0f,  0.0f, -1.0f, 0.0f, 0.0f, //  6
        size.x, -0.03f,   0.0f,  0.0f,  0.0f, -1.0f, 0.0f, 0.0f, //  7
        0.0f,     0.0f, size.y,  0.0f,  0.0f,  1.0f, 0.0f, 0.0f, //  8
        size.x,   0.0f, size.y,  0.0f,  0.0f,  1.0f, 0.0f, 0.0f, //  9
        0.0f,   -0.03f, size.y,  0.0f,  0.0f,  1.0f, 0.0f, 0.0f, // 10
        size.x, -0.03f, size.y,  0.0f,  0.0f,  1.0f, 0.0f, 0.0f, // 11
        0.0f,     0.0f,   0.0f, -1.0f,  0.0f,  0.0f, 0.0f, 0.0f, // 12
        0.0f,     0.0f, size.y, -1.0f,  0.0f,  0.0f, 0.0f, 0.0f, // 13
        0.0f,   -0.03f,   0.0f, -1.0f,  0.0f,  0.0f, 0.0f, 0.0f, // 14
        0.0f,   -0.03f, size.y, -1.0f,  0.0f,  0.0f, 0.0f, 0.0f, // 15
        size.x,   0.0f,   0.0f,  1.0f,  0.0f,  0.0f, 0.0f, 0.0f, // 16
        size.x,   0.0f, size.y,  1.0f,  0.0f,  0.0f, 0.0f, 0.0f, // 17
        size.x, -0.03f,   0.0f,  1.0f,  0.0f,  0.0f, 0.0f, 0.0f, // 18
        size.x, -0.03f, size.y,  1.0f,  0.0f,  0.0f, 0.0f, 0.0f, // 19
        0.0f,   -0.03f,   0.0f,  0.0f, -1.0f,  0.0f, 0.0f, 0.0f, // 20
        size.x, -0.03f,   0.0f,  0.0f, -1.0f,  0.0f, 0.0f, 0.0f, // 21
        0.0f,   -0.03f, size.y,  0.0f, -1.0f,  0.0f, 0.0f, 0.0f, // 22
        size.x, -0.03f, size.y,  0.0f, -1.0f,  0.0f, 0.0f, 0.0f, // 23
    };

    /*
     ,---> x
     |         4 ---------- 5
     v     12 0 ---------- 1|   16
     z    / | |6 . . . . . |7  / |
        14  | |            | 18  |
        |  13 2 ---------- 3 |  17
        | /  8 ---------- 9  | /
        15   |            |  19
             10 -------- 11
    */

    const GLuint table_indices[] = {
         0,  1,  2,
         1,  2,  3,
         4,  5,  6,
         5,  6,  7,
         8,  9, 10,
         9, 10, 11,
        12, 13, 14,
        13, 14, 15,
        16, 17, 18,
        17, 18, 19,
        20, 21, 22,
        21, 22, 23
    };

    mesh_library->set_mesh(TABLE, load_mesh("table", table_vertices,
        table_indices, 3 * 12,
        "scene/table/table_color.jpg",
        "scene/table/table_normal.jpg"));
}


//-----------------------------------------------------------------------------


/// Loads scene from a file.
Scene *load_scene(std::string file_name) {
    std::ifstream file(file_name);

    // Check if save file exists.
    if (!file) {
        LOG_ERROR("Failed to open save file!");
        return nullptr;
    }

    unsigned char index = 1;
    Scene *scene = new Scene();

    // Pointers to the most recently added sauver and vase node.
    Saucer *saucer = nullptr;
    Vase *vase = nullptr;

    // Iterate through each line.
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);

        char type;
        if (iss >> type) {
            // Table.
            if (type == 't') {
                std::string name;
                float x, y;
                if (iss >> x >> y) {
                    std::getline(iss >> std::ws, name);
                    scene->set_table(new Table(name, x, y, index++));
                } else {
                    LOG_ERROR("Failed to parse table node line!");
                    return nullptr;
                }

            // Saucer.
            } else if (type == 's') {
                std::string name;
                float x, y;
                if (iss >> x >> y) {
                    std::getline(iss >> std::ws, name);
                    saucer = new Saucer(name, scene->get_table(), x, y, index++);
                } else {
                    LOG_ERROR("Failed to parse saucer node line!");
                    return nullptr;
                }

            // Lamp.
            } else if (type == 'l') {
                std::string name;
                float x;
                if (iss >> x) {
                    std::getline(iss >> std::ws, name);
                    new Lamp(name, scene->get_table(), x, index++);
                } else {
                    LOG_ERROR("Failed to parse lamp node line!");
                    return nullptr;
                }

            // Vase.
            } else if (type == 'v') {
                std::string name;
                std::getline(iss >> std::ws, name);
                vase = new Vase(name, saucer, index++);

            // Flower
            } else if (type == 'f') {
                std::string name;
                std::getline(iss >> std::ws, name);
                new Flower(name, vase, index++);

            // Butterfly
            } else if (type == 'b') {
                std::string name;
                std::getline(iss >> std::ws, name);
                new Butterfly(name, scene->get_table(),
                    scene->get_table()->size.x * 0.5f,
                    scene->get_table()->size.y * 0.5f, index++);

            } else {
                LOG_ERROR("Failed to parse unknown line!");
                return nullptr;
            }
        }
    }

    file.close();

    // Camera.
    scene->get_camera()->view_center =
        glm::vec3(scene->get_table()->size.x * 0.5f,
            0.0f, scene->get_table()->size.y * 0.5f);
    scene->get_camera()->position = scene->get_camera()->view_center +
        glm::vec3(0.0f, 1.0f, 1.0f);
    scene->get_camera()->config = glm::vec3(0.0f, 45.0f, 1.0f);
    scene->get_camera()->is_fixed = true;
    scene->get_camera()->is_active = true;

    // Skybox.

    return scene;
}


//-----------------------------------------------------------------------------
