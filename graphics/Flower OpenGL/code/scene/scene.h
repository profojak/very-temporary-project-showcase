//-----------------------------------------------------------------------------
/**
* \file   scene.h
* \author Jakub Profota
* \brief  Scene.
*
* This file contains scene hierarchy logic and scene loader. The root of the
* scene is the table node, which holds a number of saucers, butterflies and a
* single lamp. Each saucer can then hold a single vase node, which holds a
* number of flowers.
*/
//-----------------------------------------------------------------------------


#ifndef SCENE_H
#define SCENE_H


#include "pgr.h"

#include "node.h"
#include "light.h"


/// Class that holds information about camera.
class Camera {
public:
    /// Constructor.
    Camera(void) {
    }


    /// Destructor.
    ~Camera(void) {
    }


    glm::vec3 position;    ///< Position.
    glm::vec3 view_center; ///< Center of view.
    glm::vec3 config;      ///< Longitude, latitude, distance.
    bool is_fixed;         ///< Whether the camera is fixed to the center.
    int frame_counter;     ///< Frame counter for seamless camera transition.
    glm::ivec2 resolution; ///< Window frame resolution.

    bool is_active;              ///< Whether the user can control the camera.
    glm::vec3 delta_position;    ///< Step to position to seamlessly move to.
    glm::vec3 delta_view_center; ///< Step to view center to seamlessly look at.
};


/// Class that holds information about skybox.
class Skybox {
public:
    /// Constructor.
    Skybox(void);


    /// Destructor.
    ~Skybox(void);


    GLuint vao; ///< Vertex array object.
    GLuint vbo; ///< Vertex buffer object.

    GLuint texture_cubemap; ///< Cubemap texture.
};


//-----------------------------------------------------------------------------


/// Class that holds information about scene.
class Scene {
public:
    /// Constructor.
    Scene(void);


    /// Destructor.
    ~Scene(void);


    /// Outputs the scene hierarchy to standard output.
    void print(void);


    /// Updates the scene.
    void update(void);


    /// Draws the scene on the screen.
    void draw(void);


    /// Mouse selection.
    Node *select(unsigned char index, int action);


    /// Sets table node.
    /**
     * \param[in] table Set table node.
     */
    void set_table(Table* table);


    /// Gets table node.
    Table* get_table(void) {
        return this->table;
    }


    /// Gets camera.
    Camera* get_camera(void) {
        return this->camera;
    }


    bool is_spotlight;  ///< Whether the spotlight reflector light is on.
    Light *directional; ///< Directional light.
    Light *spotlight;   ///< Spotlight light.
    Light *point;       ///< Point light.


private:
    Table *table;   ///< Pointer to table node, root of scene tree.
    Camera *camera; ///< Pointer to camera.
    Skybox *skybox; ///< Pointer to skybox.

    glm::mat4 P; ///< Perspective projection matrix.
    glm::mat4 V; ///< View transformation matrix.


    /// Rotates camera.
    void camera_rotate(void);


    /// Moves camera.
    void camera_move(void);
};


//-----------------------------------------------------------------------------


/// Loads scene from the file.
/**
* This function opens the file, parses the contents and loads the scene. The
* first entry of a valid save file starts with a table entry with dimensions.
* Saucers, their name and location on the table is specified. Each saucer can
* have a vase, and each vase can hold a number of flowers. Finally, the table
* can hold one lamp. The lamp is placed at the specified location at one of the
* table edges.
* 
* \param[in] file_name Name of the save file to parse.
* \return              Pointer to the parsed scene.
*/
Scene *load_scene(std::string file_name);


//-----------------------------------------------------------------------------


#endif //!SCENE_H
