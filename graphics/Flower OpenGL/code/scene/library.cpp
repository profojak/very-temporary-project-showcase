//-----------------------------------------------------------------------------
/**
* \file   library.h
* \author Jakub Profota
* \brief  Libraries.
*
* This file contains mesh and shader libraries.
*/
//-----------------------------------------------------------------------------


#include "library.h"

#include "../global.h"
#include "../../scene/other/other.h"


/// Constructor.
ShaderLibrary::ShaderLibrary(void) {
    this->shaders = std::vector<Shader *>(SHADER_COUNT);

    string_to_enum["flower"] = SHADER_FLOWER;
    string_to_enum["vase"] = SHADER_VASE;
    string_to_enum["saucer"] = SHADER_SAUCER;
    string_to_enum["table"] = SHADER_TABLE;
    string_to_enum["skybox"] = SHADER_SKYBOX;
    string_to_enum["butterfly"] = SHADER_BUTTERFLY;
    string_to_enum["bug"] = SHADER_BUG;
    string_to_enum["spider"] = SHADER_SPIDER;
}


/// Destructor.
ShaderLibrary::~ShaderLibrary(void) {
    for (int i = 0; i < SHADER_COUNT; i++) {
        free(shaders.at(i));
    }
}


/// Outputs the shader library to standard output.
void ShaderLibrary::print(void) {
    LOG_INFO("");
    for (int i = 0; i < SHADER_COUNT; i++) {
        if (shaders.at(i) == nullptr) {
            LOG_ERROR("Shader library is not complete!");
            return;
        }
        shaders.at(i)->print();
    }
}


//-----------------------------------------------------------------------------


/// Constructor.
MeshLibrary::MeshLibrary(void) {
    this->meshes = std::vector<Mesh *>(MESH_COUNT);
    this->colors = std::vector<glm::vec3>(COLOR_COUNT);

    string_to_enum["bergenia"] = FLOWER_BERGENIA;
    string_to_enum["crocus"] = FLOWER_CROCUS;
    string_to_enum["geranium"] = FLOWER_GERANIUM;
    string_to_enum["hibiscous"] = FLOWER_HIBISCOUS;
    string_to_enum["lilies"] = FLOWER_LILIES;
    string_to_enum["rose"] = FLOWER_ROSE;
    string_to_enum["sunflower"] = FLOWER_SUNFLOWER;
    string_to_enum["bronze"] = VASE_BRONZE;
    string_to_enum["egypt"] = VASE_EGYPT;
    string_to_enum["japan"] = VASE_JAPAN;
    string_to_enum["porcelain"] = VASE_PORCELAIN;
    string_to_enum["terracota"] = VASE_TERRACOTA;
    string_to_enum["saucer"] = SAUCER;
    string_to_enum["butterfly"] = BUTTERFLY;
    string_to_enum["bug"] = BUG;
    string_to_enum["spider"] = SPIDER;
    string_to_enum["table"] = TABLE;

    string_to_color["white"] = COLOR_WHITE;
    string_to_color["gray"] = COLOR_GRAY;
    string_to_color["black"] = COLOR_BLACK;
    string_to_color["red"] = COLOR_RED;
    string_to_color["green"] = COLOR_GREEN;
    string_to_color["blue"] = COLOR_BLUE;
    string_to_color["yellow"] = COLOR_YELLOW;
    string_to_color["cyan"] = COLOR_CYAN;
    string_to_color["magenta"] = COLOR_MAGENTA;

    colors.at(COLOR_WHITE) = glm::vec3(1.0f, 1.0f, 1.0f);
    colors.at(COLOR_GRAY) = glm::vec3(0.5f, 0.5f, 0.5f);
    colors.at(COLOR_BLACK) = glm::vec3(0.0f, 0.0f, 0.0f);
    colors.at(COLOR_RED) = glm::vec3(1.0f, 0.0f, 0.0f);
    colors.at(COLOR_GREEN) = glm::vec3(0.0f, 1.0f, 0.0f);
    colors.at(COLOR_BLUE) = glm::vec3(0.0f, 0.0f, 1.0f);
    colors.at(COLOR_YELLOW) = glm::vec3(1.0f, 1.0f, 0.0f);
    colors.at(COLOR_CYAN) = glm::vec3(0.0f, 1.0f, 1.0f);
    colors.at(COLOR_MAGENTA) = glm::vec3(1.0f, 0.0f, 1.0f);
}


/// Destructor.
MeshLibrary::~MeshLibrary(void) {
    for (int i = 0; i < MESH_COUNT; i++) {
        free(meshes.at(i));
    }
}


/// Outputs the mesh library to standard output.
void MeshLibrary::print(void) {
    LOG_INFO("");
    for (int i = 0; i < MESH_COUNT; i++) {
        if (meshes.at(i) == nullptr) {
            LOG_ERROR("Mesh library is not complete!");
            return;
        }
        meshes.at(i)->print();
    }
}


//-----------------------------------------------------------------------------


/// Loads all shaders.
ShaderLibrary* load_shader_library(void) {
    ShaderLibrary *shader_library = new ShaderLibrary();

    shader_library->set_shader(SHADER_FLOWER, load_shader("flower"));
    shader_library->set_shader(SHADER_VASE, load_shader("vase"));
    shader_library->set_shader(SHADER_SAUCER, load_shader("saucer"));
    shader_library->set_shader(SHADER_TABLE, load_shader("table"));
    shader_library->set_shader(SHADER_SKYBOX, load_shader("skybox"));
    shader_library->set_shader(SHADER_BUTTERFLY, load_shader("butterfly"));
    shader_library->set_shader(SHADER_BUG, load_shader("bug"));
    shader_library->set_shader(SHADER_SPIDER, load_shader("spider"));

    return shader_library;
}


/// Loads all meshes.
MeshLibrary* load_mesh_library(void) {
    MeshLibrary *mesh_library = new MeshLibrary();

    // Flowers.
    mesh_library->set_mesh(FLOWER_BERGENIA,
        load_mesh("scene/flower/bergenia/bergenia.obj",
            "scene/flower/bergenia/bergenia_color.jpeg",
            "scene/flower/bergenia/bergenia_normal.jpeg"));
    mesh_library->set_mesh(FLOWER_CROCUS,
        load_mesh("scene/flower/crocus/crocus.obj",
            "scene/flower/crocus/crocus_color.jpg",
            std::string()));
    mesh_library->set_mesh(FLOWER_GERANIUM,
        load_mesh("scene/flower/geranium/geranium.obj",
            "scene/flower/geranium/geranium_color.png",
            "scene/flower/geranium/geranium_normal.png"));
    mesh_library->set_mesh(FLOWER_HIBISCOUS,
        load_mesh("scene/flower/hibiscous/hibiscous.obj",
            "scene/flower/hibiscous/hibiscous_color.jpg",
            std::string()));
    mesh_library->set_mesh(FLOWER_LILIES,
        load_mesh("scene/flower/lilies/lilies.obj",
            "scene/flower/lilies/lilies_color.png",
            "scene/flower/lilies/lilies_normal.png"));
    mesh_library->set_mesh(FLOWER_ROSE,
        load_mesh("scene/flower/rose/rose.obj",
            "scene/flower/rose/rose_color.jpg",
            "scene/flower/rose/rose_normal.jpg"));
    mesh_library->set_mesh(FLOWER_SUNFLOWER,
        load_mesh("scene/flower/sunflower/sunflower.obj",
            "scene/flower/sunflower/sunflower_color.png",
            std::string()));

    // Vases.
    mesh_library->set_mesh(VASE_BRONZE,
        load_mesh("scene/vase/bronze/bronze.obj",
            "scene/vase/bronze/bronze_color.jpg",
            "scene/vase/bronze/bronze_normal.jpg"));
    mesh_library->set_mesh(VASE_EGYPT,
        load_mesh("scene/vase/egypt/egypt.obj",
            "scene/vase/egypt/egypt_color.jpg",
            std::string()));
    mesh_library->set_mesh(VASE_JAPAN,
        load_mesh("scene/vase/japan/japan.obj",
            "scene/vase/japan/japan_color.jpg",
            "scene/vase/japan/japan_normal.png"));
    mesh_library->set_mesh(VASE_PORCELAIN,
        load_mesh("scene/vase/porcelain/porcelain.obj",
            "scene/vase/porcelain/porcelain_color.jpg",
            std::string()));
    mesh_library->set_mesh(VASE_TERRACOTA,
        load_mesh("scene/vase/terracota/terracota.obj",
            "scene/vase/terracota/terracota_color.jpg",
            std::string()));

    // Saucer.
    mesh_library->set_mesh(SAUCER,
        load_mesh(std::string("saucer"),
            saucer_vertices, saucer_indices, 3 * 32,
            std::string(), std::string()));

    // Butterfly.
    mesh_library->set_mesh(BUTTERFLY,
        load_mesh(std::string("butterfly"),
            butterfly_vertices, butterfly_indices, 3 * 4,
            "scene/other/butterfly.png",
            std::string()));

    // Bug.
    mesh_library->set_mesh(BUG,
        load_mesh(std::string("bug"),
            bug_vertices, bug_indices, 3 * 2,
            "scene/other/bug.png",
            std::string()));

    // Spider.
    mesh_library->set_mesh(SPIDER,
        load_mesh(std::string("spider"),
            spider_vertices, spider_indices, 3 * 2,
            "scene/other/spider.png",
            std::string()));

    return mesh_library;
}


//-----------------------------------------------------------------------------
