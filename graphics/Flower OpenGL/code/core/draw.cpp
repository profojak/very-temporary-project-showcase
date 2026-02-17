//-----------------------------------------------------------------------------
/**
* \file   draw.cpp
* \author Jakub Profota
* \brief  Scene and node draws.
*
* This file contains scene and node draw methods.
*/
//-----------------------------------------------------------------------------


#include "pgr.h"

#include "../global.h"
#include "../scene/node.h"
#include "../scene/scene.h"


GLint elapsed = 0;


/// Sets lights uniforms.
void set_lights(Shader *shader) {
    glUniform3f(shader->get_uniform(DIRECTIONAL_AMBIENT),
        scene->directional->ambient.x,
        scene->directional->ambient.y,
        scene->directional->ambient.z);
    glUniform3f(shader->get_uniform(DIRECTIONAL_DIFFUSE),
        scene->directional->diffuse.x,
        scene->directional->diffuse.y,
        scene->directional->diffuse.z);
    glUniform3f(shader->get_uniform(DIRECTIONAL_SPECULAR),
        scene->directional->specular.x,
        scene->directional->specular.y,
        scene->directional->specular.z);
    glUniform3f(shader->get_uniform(DIRECTIONAL_DIRECTION),
        scene->directional->direction.x,
        scene->directional->direction.y,
        scene->directional->direction.z);

    glUniform1i(shader->get_uniform(SPOTLIGHT_ON), (int)scene->is_spotlight);
    glUniform3f(shader->get_uniform(SPOTLIGHT_AMBIENT),
        scene->spotlight->ambient.x,
        scene->spotlight->ambient.y,
        scene->spotlight->ambient.z);
    glUniform3f(shader->get_uniform(SPOTLIGHT_DIFFUSE),
        scene->spotlight->diffuse.x,
        scene->spotlight->diffuse.y,
        scene->spotlight->diffuse.z);
    glUniform3f(shader->get_uniform(SPOTLIGHT_SPECULAR),
        scene->spotlight->specular.x,
        scene->spotlight->specular.y,
        scene->spotlight->specular.z);
    glUniform3f(shader->get_uniform(SPOTLIGHT_POSITION),
        scene->spotlight->position.x,
        scene->spotlight->position.y,
        scene->spotlight->position.z);
    glUniform3f(shader->get_uniform(SPOTLIGHT_DIRECTION),
        scene->spotlight->direction.x,
        scene->spotlight->direction.y,
        scene->spotlight->direction.z);
    glUniform1f(shader->get_uniform(SPOTLIGHT_CONE),
        scene->spotlight->cone);
    glUniform1f(shader->get_uniform(SPOTLIGHT_EXPONENT),
        scene->spotlight->exponent);

    glUniform3f(shader->get_uniform(POINT_AMBIENT),
        scene->point->ambient.x,
        scene->point->ambient.y,
        scene->point->ambient.z);
    glUniform3f(shader->get_uniform(POINT_DIFFUSE),
        scene->point->diffuse.x,
        scene->point->diffuse.y,
        scene->point->diffuse.z);
    glUniform3f(shader->get_uniform(POINT_SPECULAR),
        scene->point->specular.x,
        scene->point->specular.y,
        scene->point->specular.z);
    glUniform3f(shader->get_uniform(POINT_POSITION),
        scene->point->position.x,
        scene->point->position.y,
        scene->point->position.z);
    glUniform1f(shader->get_uniform(POINT_EXPONENT),
        scene->point->exponent);
}


//-----------------------------------------------------------------------------


/// Draws the scene on the screen.
void Scene::draw(void) {
    elapsed++;

    // Camera.
    if (!camera->is_active) {
        camera->position += camera->delta_position;
        camera->view_center = camera->delta_view_center;

        spotlight->position = camera->position;
        spotlight->direction = glm::normalize(camera->view_center -
            camera->position);

        camera->frame_counter--;
        if (camera->frame_counter == 0) {
            camera->is_active = true;
        }
    }

    // Transformations.
    glm::mat4 V = glm::lookAt(camera->position, camera->view_center,
        glm::vec3(0.0f, 1.0f, 0.0f));
    glm::mat4 P = glm::perspective(glm::radians(60.0f),
        (float)camera->resolution.x / (float)camera->resolution.y,
        0.1f, 10.0f);

    // Skybox.
    glDepthMask(GL_FALSE);
    Shader *shader = shader_library->get_shader("skybox");
    glUseProgram(shader->get_id());

    glUniformMatrix4fv(shader->get_uniform(UNIFORM_P), 1, GL_FALSE,
        glm::value_ptr(P));
    CHECK_GL_ERROR();
    glUniformMatrix4fv(shader->get_uniform(UNIFORM_V), 1, GL_FALSE,
        glm::value_ptr(glm::mat4(glm::mat3(V))));
    CHECK_GL_ERROR();

    glUniform1i(shader->get_uniform(SKYBOX), 0);
    CHECK_GL_ERROR();
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_CUBE_MAP, skybox->texture_cubemap);

    glBindVertexArray(skybox->vao);
    glDrawArrays(GL_TRIANGLES, 0, 36);
    glDepthMask(GL_TRUE);

    glBindVertexArray(0);
    glUseProgram(0);

    // Table.
    this->table->draw(P, V);
}


//-----------------------------------------------------------------------------


/// Draws the node on the screen.
void Table::draw(glm::mat4 &P, glm::mat4 &V) {
    // Transformations.
    glm::mat4 M = glm::translate(glm::mat4(1.0f), glm::vec3(
        global_position.x, 0.0f, global_position.y));

    // Draw.
    Shader *shader = shader_library->get_shader("table");
    Mesh *mesh = mesh_library->get_mesh("table");

    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    glUseProgram(shader->get_id());

    glUniformMatrix4fv(shader->get_uniform(UNIFORM_P), 1, GL_FALSE,
        glm::value_ptr(P));
    CHECK_GL_ERROR();
    glUniformMatrix4fv(shader->get_uniform(UNIFORM_V), 1, GL_FALSE,
        glm::value_ptr(V));
    CHECK_GL_ERROR();
    glUniformMatrix4fv(shader->get_uniform(UNIFORM_M), 1, GL_FALSE,
        glm::value_ptr(M));
    CHECK_GL_ERROR();

    glUniform1i(shader->get_uniform(TEXTURE_COLOR), 0);
    CHECK_GL_ERROR();
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, mesh->get_texture_color());

    glUniform3f(shader->get_uniform(MATERIAL_AMBIENT), 1.0f, 1.0f, 0.9f);
    CHECK_GL_ERROR();
    glUniform3f(shader->get_uniform(MATERIAL_DIFFUSE), 1.0f, 1.0f, 0.9f);
    CHECK_GL_ERROR();
    glUniform3f(shader->get_uniform(MATERIAL_SPECULAR), 1.0f, 1.0f, 1.0f);
    CHECK_GL_ERROR();
    glUniform1f(shader->get_uniform(MATERIAL_SHININESS), 2.0f);
    CHECK_GL_ERROR();

    set_lights(shader);

    glBindVertexArray(mesh->get_vao());
    glDrawElements(GL_TRIANGLES, 3 * mesh->get_n_triangles(),
        GL_UNSIGNED_INT, (void *)0);

    glBindVertexArray(0);
    glUseProgram(0);

    // Saucers.
    for (std::vector<Saucer *>::iterator i = saucers.begin();
        i != saucers.end(); ++i) {
        (*i)->draw(P, V);
    }

    // Butterflies.
    for (std::vector<Butterfly *>::iterator i = butterflies.begin();
        i != butterflies.end(); ++i) {
        (*i)->draw(P, V);
    }

    // Bug.
    this->bug->draw(P, V);

    // Spider.
    this->spider->draw(P, V);

    // Lamp.
    if (this->lamp) {
        this->lamp->draw(P, V);
    }
}


/// Draws the node on the screen.
void Saucer::draw(glm::mat4 &P, glm::mat4 &V) {
    // Transformations.
    glm::mat4 M = glm::translate(glm::mat4(1.0f), glm::vec3(
        global_position.x, 0.0f, global_position.y));

    // Draw.
    Shader *shader = shader_library->get_shader("saucer");
    Mesh *mesh = mesh_library->get_mesh("saucer");
    glm::vec3 color = mesh_library->get_color(name);

    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    glEnable(GL_STENCIL_TEST);
    glStencilOp(GL_KEEP, GL_KEEP, GL_REPLACE);
    glUseProgram(shader->get_id());

    glStencilFunc(GL_ALWAYS, index, 0);

    glUniformMatrix4fv(shader->get_uniform(UNIFORM_P), 1, GL_FALSE,
        glm::value_ptr(P));
    CHECK_GL_ERROR();
    glUniformMatrix4fv(shader->get_uniform(UNIFORM_V), 1, GL_FALSE,
        glm::value_ptr(V));
    CHECK_GL_ERROR();
    glUniformMatrix4fv(shader->get_uniform(UNIFORM_M), 1, GL_FALSE,
        glm::value_ptr(M));
    CHECK_GL_ERROR();

    glUniform3f(shader->get_uniform(UNIFORM_COLOR),
        color.x, color.y, color.z);
    CHECK_GL_ERROR();

    set_lights(shader);

    glBindVertexArray(mesh->get_vao());
    glDrawElements(GL_TRIANGLES, 3 * mesh->get_n_triangles(),
        GL_UNSIGNED_INT, (void *)0);

    glUniform3f(shader->get_uniform(MATERIAL_AMBIENT), 1.0f, 1.0f, 1.0f);
    CHECK_GL_ERROR();
    glUniform3f(shader->get_uniform(MATERIAL_DIFFUSE), 1.0f, 1.0f, 1.0f);
    CHECK_GL_ERROR();
    glUniform3f(shader->get_uniform(MATERIAL_SPECULAR), 1.0f, 1.0f, 1.0f);
    CHECK_GL_ERROR();
    glUniform1f(shader->get_uniform(MATERIAL_SHININESS), 1.0f);
    CHECK_GL_ERROR();

    glBindVertexArray(0);
    glUseProgram(0);
    glDisable(GL_STENCIL_TEST);

    // Vase.
    if (this->vase) {
        this->vase->draw(P, V);
    }
}


/// Draws the node on the screen.
void Vase::draw(glm::mat4 &P, glm::mat4 &V) {
    // Transformations.
    glm::mat4 M = glm::translate(glm::mat4(1.0f), glm::vec3(
        global_position.x, 0.0f, global_position.y));

    // Draw.
    Shader *shader = shader_library->get_shader("vase");
    Mesh *mesh = mesh_library->get_mesh(name);

    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    glEnable(GL_STENCIL_TEST);
    glStencilOp(GL_KEEP, GL_KEEP, GL_REPLACE);
    glUseProgram(shader->get_id());

    glStencilFunc(GL_ALWAYS, index, 0);

    glUniformMatrix4fv(shader->get_uniform(UNIFORM_P), 1, GL_FALSE,
        glm::value_ptr(P));
    CHECK_GL_ERROR();
    glUniformMatrix4fv(shader->get_uniform(UNIFORM_V), 1, GL_FALSE,
        glm::value_ptr(V));
    CHECK_GL_ERROR();
    glUniformMatrix4fv(shader->get_uniform(UNIFORM_M), 1, GL_FALSE,
        glm::value_ptr(M));
    CHECK_GL_ERROR();

    glUniform1i(shader->get_uniform(TEXTURE_COLOR), 0);
    CHECK_GL_ERROR();
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, mesh->get_texture_color());

    glUniform3f(shader->get_uniform(MATERIAL_AMBIENT), 0.9f, 0.9f, 1.0f);
    CHECK_GL_ERROR();
    glUniform3f(shader->get_uniform(MATERIAL_DIFFUSE), 0.9f, 0.9f, 1.0f);
    CHECK_GL_ERROR();
    glUniform3f(shader->get_uniform(MATERIAL_SPECULAR), 0.1f, 0.1f, 0.15f);
    CHECK_GL_ERROR();
    glUniform1f(shader->get_uniform(MATERIAL_SHININESS), 3.0f);
    CHECK_GL_ERROR();

    set_lights(shader);

    glBindVertexArray(mesh->get_vao());
    glDrawElements(GL_TRIANGLES, 3 * mesh->get_n_triangles(),
        GL_UNSIGNED_INT, (void *)0);

    glBindVertexArray(0);
    glUseProgram(0);
    glDisable(GL_STENCIL_TEST);

    // Flowers.
    for (std::vector<Flower *>::iterator i = flowers.begin();
        i != flowers.end(); ++i) {
        (*i)->draw(P, V);
    }
}


/// Draws the node on the screen.
void Flower::draw(glm::mat4 &P, glm::mat4 &V) {
    // Transformations.
    glm::mat4 M = glm::translate(glm::mat4(1.0f), glm::vec3(
        global_position.x, 0.0f, global_position.y));

    // Draw.
    Shader *shader = shader_library->get_shader("flower");
    Mesh *mesh = mesh_library->get_mesh(name);

    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    glEnable(GL_STENCIL_TEST);
    glStencilOp(GL_KEEP, GL_KEEP, GL_REPLACE);
    glUseProgram(shader->get_id());

    glStencilFunc(GL_ALWAYS, index, 0);

    glUniformMatrix4fv(shader->get_uniform(UNIFORM_P), 1, GL_FALSE,
        glm::value_ptr(P));
    CHECK_GL_ERROR();
    glUniformMatrix4fv(shader->get_uniform(UNIFORM_V), 1, GL_FALSE,
        glm::value_ptr(V));
    CHECK_GL_ERROR();
    glUniformMatrix4fv(shader->get_uniform(UNIFORM_M), 1, GL_FALSE,
        glm::value_ptr(M));
    CHECK_GL_ERROR();
    glUniform1i(shader->get_uniform(UNIFORM_ELAPSED), elapsed);

    glUniform1i(shader->get_uniform(TEXTURE_COLOR), 0);
    CHECK_GL_ERROR();
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, mesh->get_texture_color());

    glUniform3f(shader->get_uniform(MATERIAL_AMBIENT), 1.0f, 1.0f, 1.0f);
    CHECK_GL_ERROR();
    glUniform3f(shader->get_uniform(MATERIAL_DIFFUSE), 1.0f, 1.0f, 1.0f);
    CHECK_GL_ERROR();
    glUniform3f(shader->get_uniform(MATERIAL_SPECULAR), 1.0f, 1.0f, 1.0f);
    CHECK_GL_ERROR();
    glUniform1f(shader->get_uniform(MATERIAL_SHININESS), 1.0f);
    CHECK_GL_ERROR();

    set_lights(shader);

    glBindVertexArray(mesh->get_vao());
    glDrawElements(GL_TRIANGLES, 3 * mesh->get_n_triangles(),
        GL_UNSIGNED_INT, (void *)0);

    glBindVertexArray(0);
    glUseProgram(0);
    glDisable(GL_STENCIL_TEST);
}


/// Draws the node on the screen.
void Butterfly::draw(glm::mat4 &P, glm::mat4 &V) {
    // Transformations.
    glm::vec2 size = scene->get_table()->size;
    float x = global_position.x + sin(0.01f * (float)elapsed) * size.x * 0.5f;
    float y = global_position.y + cos(0.01f * (float)elapsed) * size.y * 0.5f;
    glm::mat4 M = glm::translate(glm::mat4(1.0f), glm::vec3(x, 0.3f, y));
    M = glm::rotate(M, atan2(x - global_position.x, y - global_position.y) +
        glm::radians(-90.0f), glm::vec3(0.0f, 1.0f, 0.0f));

    glm::vec4 pos = M * glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);

    scene->point->position.x = pos.x;
    scene->point->position.y = pos.y;
    scene->point->position.z = pos.z;

    // Draw.
    Shader *shader = shader_library->get_shader("butterfly");
    Mesh *mesh = mesh_library->get_mesh("butterfly");

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glUseProgram(shader->get_id());

    glUniformMatrix4fv(shader->get_uniform(UNIFORM_P), 1, GL_FALSE,
        glm::value_ptr(P));
    CHECK_GL_ERROR();
    glUniformMatrix4fv(shader->get_uniform(UNIFORM_V), 1, GL_FALSE,
        glm::value_ptr(V));
    CHECK_GL_ERROR();
    glUniformMatrix4fv(shader->get_uniform(UNIFORM_M), 1, GL_FALSE,
        glm::value_ptr(M));
    CHECK_GL_ERROR();
    glUniform1i(shader->get_uniform(UNIFORM_ELAPSED), elapsed);

    glUniform1i(shader->get_uniform(TEXTURE_COLOR), 0);
    CHECK_GL_ERROR();
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, mesh->get_texture_color());

    glBindVertexArray(mesh->get_vao());
    glDrawElements(GL_TRIANGLES, 3 * mesh->get_n_triangles(),
        GL_UNSIGNED_INT, (void *)0);

    glBindVertexArray(0);
    glUseProgram(0);
    glDisable(GL_BLEND);
}


/// Draws the node on the screen.
void Bug::draw(glm::mat4 &P, glm::mat4 &V) {
    // Transformations.
    glm::vec2 size = scene->get_table()->size;
    glm::mat4 M = glm::translate(glm::mat4(1.0f),
        glm::vec3(size.x * 0.3f, 0.0f, size.y - 0.05f));

    // Draw.
    Shader *shader = shader_library->get_shader("bug");
    Mesh *mesh = mesh_library->get_mesh("bug");
    glBindTexture(GL_TEXTURE_2D, mesh->get_texture_color());
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glUseProgram(shader->get_id());

    glUniformMatrix4fv(shader->get_uniform(UNIFORM_P), 1, GL_FALSE,
        glm::value_ptr(P));
    CHECK_GL_ERROR();
    glUniformMatrix4fv(shader->get_uniform(UNIFORM_V), 1, GL_FALSE,
        glm::value_ptr(V));
    CHECK_GL_ERROR();
    glUniformMatrix4fv(shader->get_uniform(UNIFORM_M), 1, GL_FALSE,
        glm::value_ptr(M));
    CHECK_GL_ERROR();
    glUniform1i(shader->get_uniform(UNIFORM_ELAPSED), elapsed);

    glUniform1i(shader->get_uniform(TEXTURE_COLOR), 0);
    CHECK_GL_ERROR();
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, mesh->get_texture_color());

    glBindVertexArray(mesh->get_vao());
    glDrawElements(GL_TRIANGLES, 3 * mesh->get_n_triangles(),
        GL_UNSIGNED_INT, (void *)0);

    glBindVertexArray(0);
    glUseProgram(0);
    glDisable(GL_BLEND);
}


/// Draws the node on the screen.
void Spider::draw(glm::mat4 &P, glm::mat4 &V) {
    // Transformations.
    glm::vec2 size = scene->get_table()->size;
    glm::mat4 M = glm::translate(glm::mat4(1.0f),
        glm::vec3(0.0f, -0.03f, 0.0f));

    // Draw.
    Shader *shader = shader_library->get_shader("spider");
    Mesh *mesh = mesh_library->get_mesh("spider");
    glBindTexture(GL_TEXTURE_2D, mesh->get_texture_color());

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glUseProgram(shader->get_id());

    glUniformMatrix4fv(shader->get_uniform(UNIFORM_P), 1, GL_FALSE,
        glm::value_ptr(P));
    CHECK_GL_ERROR();
    glUniformMatrix4fv(shader->get_uniform(UNIFORM_V), 1, GL_FALSE,
        glm::value_ptr(V));
    CHECK_GL_ERROR();
    glUniformMatrix4fv(shader->get_uniform(UNIFORM_M), 1, GL_FALSE,
        glm::value_ptr(M));
    CHECK_GL_ERROR();
    glUniform1i(shader->get_uniform(UNIFORM_ELAPSED), elapsed);

    glUniform1i(shader->get_uniform(TEXTURE_COLOR), 0);
    CHECK_GL_ERROR();
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, mesh->get_texture_color());

    glBindVertexArray(mesh->get_vao());
    glDrawElements(GL_TRIANGLES, 3 * mesh->get_n_triangles(),
        GL_UNSIGNED_INT, (void *)0);

    glBindVertexArray(0);
    glUseProgram(0);
    glDisable(GL_BLEND);
}


/// Draws the node on the screen.
void Lamp::draw(glm::mat4 &P, glm::mat4 &V) {
}


//-----------------------------------------------------------------------------
