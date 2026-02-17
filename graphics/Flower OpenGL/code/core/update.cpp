//-----------------------------------------------------------------------------
/**
* \file   update.cpp
* \author Jakub Profota
* \brief  Scene and node updates.
*
* This file contains scene and node update methods.
*/
//-----------------------------------------------------------------------------


#include "pgr.h"

#include "../global.h"
#include "../platform/keyboard.h"
#include "../scene/node.h"
#include "../scene/scene.h"


/// Updates the scene.
void Scene::update(void) {
    // Camera.
    if (camera->is_fixed && camera->is_active) {
        camera_rotate();
    } else if (camera->is_active) {
        camera_move();
    }

    // Light.
    spotlight->position = camera->position;
    spotlight->direction = glm::normalize(camera->view_center - camera->position);

    // Table.
    this->table->update(glm::vec2(0.0f), glm::vec2(0.0f));
}


/// Rotates camera.
void Scene::camera_rotate(void) {
    // Rotate camera left.
    if (keyboard_map['a'] || keyboard_map['A']) {
        if (keyboard_map[KEYBOARD_LEFT_SHIFT] || keyboard_map['A']) {
            camera->config.x += 2.5f;
        } else {
            camera->config.x += 0.5f;
        }

        if (camera->config.x >= 360.f) {
            camera->config.x -= 360.0f;
        }

    // Rotate camera right.
    } else if (keyboard_map['d'] || keyboard_map['D']) {
        if (keyboard_map[KEYBOARD_LEFT_SHIFT] || keyboard_map['D']) {
            camera->config.x -= 2.5f;
        } else {
            camera->config.x -= 0.5f;
        }

        if (camera->config.x < 0.0f) {
            camera->config.x += 360.0f;
        }

    // Rotate camera up.
    } else if (keyboard_map['e'] || keyboard_map['E']) {
        if (keyboard_map[KEYBOARD_LEFT_SHIFT] || keyboard_map['E']) {
            camera->config.y -= 2.5f;
        } else {
            camera->config.y -= 0.5f;
        }

        if (camera->config.y < 0.5f) {
            camera->config.y = 0.5f;
        }

    // Rotate camera down.
    } else if (keyboard_map['q'] || keyboard_map['Q']) {
        if (keyboard_map[KEYBOARD_LEFT_SHIFT] || keyboard_map['Q']) {
            camera->config.y += 2.5f;
        } else {
            camera->config.y += 0.5f;
        }

        if (camera->config.y > 179.5f) {
            camera->config.y = 179.5f;
        }

    // Decrease camera distance.
    } else if (keyboard_map['w'] || keyboard_map['W']) {
        if (keyboard_map[KEYBOARD_LEFT_SHIFT] || keyboard_map['W']) {
            camera->config.z -= 0.025f;
        } else {
            camera->config.z -= 0.005f;
        }

        if (camera->config.z < 0.1f) {
            camera->config.z = 0.1f;
        }

    // Increase camera distance.
    } else if (keyboard_map['s'] || keyboard_map['S']) {
        if (keyboard_map[KEYBOARD_LEFT_SHIFT] || keyboard_map['S']) {
            camera->config.z += 0.025f;
        } else {
            camera->config.z += 0.005f;
        }
    }

    // Spheric coordinates.
    glm::vec2 size = get_table()->size;

    float s0, c0, s1, c1;
    s0 = sinf(glm::radians(camera->config.x));
    c0 = cosf(glm::radians(camera->config.x));
    s1 = sinf(glm::radians(camera->config.y));
    c1 = cosf(glm::radians(camera->config.y));
    camera->position.x = camera->config.z * s1 * c0 + size.x * 0.5f;
    camera->position.y = camera->config.z * c1;
    camera->position.z = camera->config.z * s1 * s0 + size.y * 0.5f;

    // Keep camera in bounds.
    if (camera->position.x < -2.0f || camera->position.x > size.x + 2.0f ||
        camera->position.y < -2.0f || camera->position.y > 2.0f ||
        camera->position.z < -2.0f || camera->position.z > size.y + 2.0f) {
        if (keyboard_map['a'] || keyboard_map['A']) {
            if (keyboard_map[KEYBOARD_LEFT_SHIFT] || keyboard_map['A']) {
                camera->config.x -= 2.5f;
            } else {
                camera->config.x -= 0.5f;
            }
        } else if (keyboard_map['d'] || keyboard_map['D']) {
            if (keyboard_map[KEYBOARD_LEFT_SHIFT] || keyboard_map['D']) {
                camera->config.x += 2.5f;
            } else {
                camera->config.x += 0.5f;
            }
        } else if (keyboard_map['e'] || keyboard_map['E']) {
            if (keyboard_map[KEYBOARD_LEFT_SHIFT] || keyboard_map['E']) {
                camera->config.y += 2.5f;
            } else {
                camera->config.y += 0.5f;
            }
        } else if (keyboard_map['q'] || keyboard_map['Q']) {
            if (keyboard_map[KEYBOARD_LEFT_SHIFT] || keyboard_map['Q']) {
                camera->config.y -= 2.5f;
            } else {
                camera->config.y -= 0.5f;
            }
        } else if (keyboard_map['w'] || keyboard_map['W']) {
            if (keyboard_map[KEYBOARD_LEFT_SHIFT] || keyboard_map['W']) {
                camera->config.z += 0.025f;
            } else {
                camera->config.z += 0.005f;
            }
        } else if (keyboard_map['s'] || keyboard_map['S']) {
            if (keyboard_map[KEYBOARD_LEFT_SHIFT] || keyboard_map['S']) {
                camera->config.z -= 0.025f;
            } else {
                camera->config.z -= 0.005f;
            }
        }

        s0 = sinf(glm::radians(camera->config.x));
        c0 = cosf(glm::radians(camera->config.x));
        s1 = sinf(glm::radians(camera->config.y));
        c1 = cosf(glm::radians(camera->config.y));
        camera->position.x = camera->config.z * s1 * c0 + size.x * 0.5f;
        camera->position.y = camera->config.z * c1;
        camera->position.z = camera->config.z * s1 * s0 + size.y * 0.5f;
    }
}


/// Moves camera.
void Scene::camera_move(void) {
    float s0, c0, s1, c1;
    s0 = sinf(glm::radians(camera->config.x));
    c0 = cosf(glm::radians(camera->config.x));
    s1 = sinf(glm::radians(camera->config.y));
    c1 = cosf(glm::radians(camera->config.y));
    camera->view_center.x = camera->config.z * s1 * c0 * 0.001f;
    camera->view_center.y = camera->config.z * c1 * 0.001f;
    camera->view_center.z = camera->config.z * s1 * s0 * 0.001f;

    // Move camera left.
    if (keyboard_map['a'] || keyboard_map['A']) {
        glm::vec3 right = glm::cross(camera->view_center,
            glm::vec3(0.0f, 1.0f, 0.0f));
        camera->position -= right;
        if (keyboard_map[KEYBOARD_LEFT_SHIFT] || keyboard_map['A']) {
            camera->position -= right;
            camera->position -= right;
        }

    // Move camera right.
    } else if (keyboard_map['d'] || keyboard_map['D']) {
        glm::vec3 right = glm::cross(camera->view_center,
            glm::vec3(0.0f, 1.0f, 0.0f));
        camera->position += right;
        if (keyboard_map[KEYBOARD_LEFT_SHIFT] || keyboard_map['D']) {
            camera->position += right;
            camera->position += right;
        }

    // Move camera up.
    } else if (keyboard_map['e'] || keyboard_map['E']) {
        glm::vec3 right = glm::normalize(glm::cross(camera->view_center,
            glm::vec3(0.0f, 1.0f, 0.0f)));
        glm::vec3 up = glm::cross(camera->view_center, right);
        camera->position -= up;
        if (keyboard_map[KEYBOARD_LEFT_SHIFT] || keyboard_map['E']) {
            camera->position -= up;
            camera->position -= up;
        }

    // Move camera down.
    } else if (keyboard_map['q'] || keyboard_map['Q']) {
        glm::vec3 right = glm::normalize(glm::cross(camera->view_center,
            glm::vec3(0.0f, 1.0f, 0.0f)));
        glm::vec3 up = glm::cross(camera->view_center, right);
        camera->position += up;
        if (keyboard_map[KEYBOARD_LEFT_SHIFT] || keyboard_map['Q']) {
            camera->position += up;
            camera->position += up;
        }

    // Move camera forward.
    } else if (keyboard_map['w'] || keyboard_map['W']) {
        camera->position += camera->view_center;
        if (keyboard_map[KEYBOARD_LEFT_SHIFT] || keyboard_map['W']) {
            camera->position += camera->view_center;
            camera->position += camera->view_center;
        }

    // Move camera backward.
    } else if (keyboard_map['s'] || keyboard_map['S']) {
        camera->position -= camera->view_center;
        if (keyboard_map[KEYBOARD_LEFT_SHIFT] || keyboard_map['S']) {
            camera->position -= camera->view_center;
            camera->position -= camera->view_center;
        }
    }

    // Keep camera in bounds.
    glm::vec2 size = get_table()->size;
    if (camera->position.x < -2.0f) {
        camera->position.x = -2.0f;
    }
    if (camera->position.x > size.x + 2.0f) {
        camera->position.x = size.x + 2.0f;
    }
    if (camera->position.y < -2.0f) {
        camera->position.y = -2.0f;
    }
    if (camera->position.y > 2.0f) {
        camera->position.y = 2.0f;
    }
    if (camera->position.z < -2.0f) {
        camera->position.z = -2.0f;
    }
    if (camera->position.z > size.y + 2.0f) {
        camera->position.z = size.y + 2.0f;
    }

    camera->view_center.x += camera->position.x;
    camera->view_center.y += camera->position.y;
    camera->view_center.z += camera->position.z;
}


//-----------------------------------------------------------------------------


/// Updates the node.
void Table::update(glm::vec2 &global_position, glm::vec2 &global_rotation) {
    this->global_position = global_position + this->local_position;
    this->global_rotation = global_rotation + this->local_rotation;

    // Saucers.
    for (std::vector<Saucer *>::iterator i = saucers.begin();
        i != saucers.end(); ++i) {
        (*i)->update(this->global_position, this->global_rotation);
    }

    // Butterflies.
    for (std::vector<Butterfly *>::iterator i = butterflies.begin();
        i != butterflies.end(); ++i) {
        (*i)->update(this->global_position, this->global_rotation);
    }

    // Lamp.
    if (this->lamp) {
        this->lamp->update(this->global_position, this->global_rotation);
    }
}


/// Updates the node.
void Saucer::update(glm::vec2 &global_position, glm::vec2 &global_rotation) {
    this->global_position = global_position + this->local_position;
    this->global_rotation = global_rotation + this->local_rotation;

    // Vase.
    if (this->vase) {
        this->vase->update(this->global_position, this->global_rotation);
    }
}


/// Updates the node.
void Vase::update(glm::vec2 &global_position, glm::vec2 &global_rotation) {
    this->global_position = global_position + this->local_position;
    this->global_rotation = global_rotation + this->local_rotation;

    // Flowers.
    for (std::vector<Flower *>::iterator i = flowers.begin();
        i != flowers.end(); ++i) {
        (*i)->update(this->global_position, this->global_rotation);
    }
}


/// Updates the node.
void Flower::update(glm::vec2 &global_position, glm::vec2 &global_rotation) {
    this->global_position = global_position + this->local_position;
    this->global_rotation = global_rotation + this->local_rotation;
}


/// Updates the node.
void Butterfly::update(glm::vec2 &global_position, glm::vec2 &global_rotation) {
    this->global_position = global_position + this->local_position;
    this->global_rotation = global_rotation + this->local_rotation;
}


/// Updates the node.
void Lamp::update(glm::vec2 &global_position, glm::vec2 &global_rotation) {
    this->global_position = global_position + this->local_position;
    this->global_rotation = global_rotation + this->local_rotation;
}


//-----------------------------------------------------------------------------
