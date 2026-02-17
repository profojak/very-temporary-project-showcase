/* CUDA C Software Rendering Pipeline ---------------------------------------
   Bachelor Thesis                                              Jakub Profota
   Summer Semester 2022/3                    OI - Computer Games and Graphics
   Supervisor Jiri Bittner               Czech Technical University in Prague
   mesh.c ------------------------------------------------------------------- */


#include "mesh.h"

#include "define.h"
#include "verbose.h"
#include "../platform/memory.h"
#include "../platform/print.h"

#include <stdio.h>  // FILE
#include <stdlib.h> // EXIT_SUCCESS, EXIT_FAILURE


/* Load and unload ---------------------------------------------------------- */


/* Load mesh */
extern int mesh_load(
    mesh_t *mesh // Mesh to load
) {
    if (global_verbose) {
        #ifdef _DEBUG
        LOG_DEBUG("Loading mesh . . .");
        #else
        print(PRINT_WHITE_FG, "Loading mesh . . .\n");
        #endif /* _DEBUG */
    }

    size_t max_size_vertices = MESH_BATCH_SIZE;
    size_t max_size_normals = MESH_BATCH_SIZE;
    size_t max_size_indices = MESH_BATCH_SIZE * 3;
    mesh->size_vertices = 0;
    mesh->size_normals = 0;
    mesh->size_indices = 0;
    mesh->vertices = (vec3_t *)mem_alloc(max_size_vertices * sizeof(vec3_t));
    mesh->normals = (vec3_t *)mem_alloc(max_size_normals * sizeof(vec3_t));
    mesh->indices = (ivec2_t *)mem_alloc(max_size_indices * sizeof(ivec2_t));

    // Open object file
    char line[1024];
    char path[1024];
    mem_set(line, 0, 1024);
    mem_set(path, 0, 1024);

    if (global_mesh == 0) {
        global_mesh = (char *)"cube";
    }

    strcat(path, "obj\\");
    strcat(path, global_mesh);
    strcat(path, ".obj");

    FILE *file = fopen(path, "r");
    if (file == 0) {
        #ifdef _DEBUG
        LOG_ERROR("Failed to open object file!");
        LOG_DEBUG("Object file \"%s.obj\" may not exist",
            global_mesh);
        LOG_DEBUG("Please run \"CUDA-pipeline.exe\" from root directory:");
        LOG_WARNING(".\\bin\\<target>\\CUDA-pipeline.exe -model=<name>");
        #endif /* _DEBUG */
        return EXIT_FAILURE;
    }

    while (fgets(line, 1023, file) != 0) {

        // Vertex
        if (line[0] == 'v' && line[1] == ' ') {
            if (mesh->size_vertices >= max_size_vertices) {
                max_size_vertices += MESH_BATCH_SIZE;
                mesh->vertices = (vec3_t *)mem_realloc(
                    (void *)mesh->vertices,
                    max_size_vertices * sizeof(vec3_t)
                );
                if (mesh->vertices == 0) {
                    #ifdef _DEBUG
                    LOG_ERROR("Failed to reallocate vertex buffer!");
                    #endif /* _DEBUG */
                    return EXIT_FAILURE;
                }
            }

            sscanf(line, "v %f %f %f",
                &mesh->vertices[mesh->size_vertices][0],
                &mesh->vertices[mesh->size_vertices][1],
                &mesh->vertices[mesh->size_vertices][2]
            );
            mesh->size_vertices++;

        // Vertex normal
        } else if (line[0] == 'v' && line[1] == 'n') {
            if (mesh->size_normals >= max_size_normals) {
                max_size_normals += MESH_BATCH_SIZE;
                mesh->normals = (vec3_t *)mem_realloc(
                    (void *)mesh->normals,
                    max_size_normals * sizeof(vec3_t)
                );
                if (mesh->normals == 0) {
                    #ifdef _DEBUG
                    LOG_ERROR("Failed to reallocate normal buffer!");
                    #endif /* _DEBUG */
                    return EXIT_FAILURE;
                }
            }

            sscanf(line, "vn %f %f %f",
                &mesh->normals[mesh->size_normals][0],
                &mesh->normals[mesh->size_normals][1],
                &mesh->normals[mesh->size_normals][2]
            );
            mesh->size_normals++;

        // Face
        } else if (line[0] == 'f' && line[1] == ' ') {
            if (mesh->size_indices >= max_size_indices) {
                max_size_indices += MESH_BATCH_SIZE * 3;
                mesh->indices = (ivec2_t *)mem_realloc(
                    (void *)mesh->indices,
                    max_size_indices * sizeof(ivec2_t)
                );
                if (mesh->indices == 0) {
                    #ifdef _DEBUG
                    LOG_ERROR("Failed to reallocate index buffer!");
                    #endif /* _DEBUG */
                    return EXIT_FAILURE;
                }
            }

            sscanf(line, "f %i//%i %i//%i %i//%i",
                &mesh->indices[mesh->size_indices + 0][0],
                &mesh->indices[mesh->size_indices + 0][1],
                &mesh->indices[mesh->size_indices + 1][0],
                &mesh->indices[mesh->size_indices + 1][1],
                &mesh->indices[mesh->size_indices + 2][0],
                &mesh->indices[mesh->size_indices + 2][1]
            );
            mesh->size_indices += 3;
        }
    }

    fclose(file);
    file = 0;

    #ifndef _DEBUG
    print(PRINT_WHITE_FG, "Loaded mesh\n");
    #else
    LOG_TEXT("Loaded mesh");
    LOG_TRACE;

    if (global_verbose) {
        verbose_mesh(mesh);
    }
    #endif /* _DEBUG */

    return EXIT_SUCCESS;
} /* mesh_load */


/* Unload mesh */
extern int mesh_unload(
    mesh_t *mesh // Mesh to unload
) {
    mem_free(mesh->vertices);
    mem_free(mesh->normals);
    mem_free(mesh->indices);

    #ifdef _DEBUG
    LOG_TEXT("Unloaded mesh");
    LOG_TRACE;
    #endif /* _DEBUG */

    return EXIT_SUCCESS;
} /* mesh_unload */


/* -------------------------------------------------------------------------- */


/* mesh.c */