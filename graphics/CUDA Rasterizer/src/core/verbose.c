/* CUDA C Software Rendering Pipeline ---------------------------------------
   Bachelor Thesis                                              Jakub Profota
   Summer Semester 2022/3                    OI - Computer Games and Graphics
   Supervisor Jiri Bittner               Czech Technical University in Prague
   verbose.c ---------------------------------------------------------------- */


#ifdef _DEBUG


#include "verbose.h"


#include "define.h"
#include "../platform/print.h"

#include <stdlib.h> // size_t


/* Vector ------------------------------------------------------------------- */


/* Print vec2_t */
extern void verbose_vec2(
    vec2_t vec // Vector
) {
    print(PRINT_LIME_FG, " (");
    print(PRINT_WHITE_FG, "%16.8f", vec[0]);
    print(PRINT_GREEN_FG, ", ");
    print(PRINT_WHITE_FG, "%16.8f", vec[1]);
    print(PRINT_LIME_FG, ")\n");
} /* verbose_vec2 */


/* Print ivec2_t */
extern void verbose_ivec2(
    ivec2_t vec // Vector
) {
    print(PRINT_LIME_FG, " (");
    print(PRINT_WHITE_FG, "%16i", vec[0]);
    print(PRINT_GREEN_FG, ", ");
    print(PRINT_WHITE_FG, "%16i", vec[1]);
    print(PRINT_LIME_FG, ")\n");
} /* verbose_ivec2 */


/* Print vec3_t */
extern void verbose_vec3(
    vec3_t vec // Vector
) {
    print(PRINT_LIME_FG, " (");
    print(PRINT_WHITE_FG, "%16.8f", vec[0]);
    print(PRINT_GREEN_FG, ", ");
    print(PRINT_WHITE_FG, "%16.8f", vec[1]);
    print(PRINT_GREEN_FG, ", ");
    print(PRINT_WHITE_FG, "%16.8f", vec[2]);
    print(PRINT_LIME_FG, ")\n");
} /* verbose_vec3 */


/* Print ivec3_t */
extern void verbose_ivec3(
    ivec3_t vec // Vector
) {
    print(PRINT_LIME_FG, " (");
    print(PRINT_WHITE_FG, "%16i", vec[0]);
    print(PRINT_GREEN_FG, ", ");
    print(PRINT_WHITE_FG, "%16i", vec[1]);
    print(PRINT_GREEN_FG, ", ");
    print(PRINT_WHITE_FG, "%16i", vec[2]);
    print(PRINT_LIME_FG, ")\n");
} /* verbose_ivec3 */


/* Print vec4_t */
extern void verbose_vec4(
    vec4_t vec // Vector
) {
    print(PRINT_LIME_FG, " (");
    print(PRINT_WHITE_FG, "%16.8f", vec[0]);
    print(PRINT_GREEN_FG, ", ");
    print(PRINT_WHITE_FG, "%16.8f", vec[1]);
    print(PRINT_GREEN_FG, ", ");
    print(PRINT_WHITE_FG, "%16.8f", vec[2]);
    print(PRINT_GREEN_FG, ", ");
    print(PRINT_WHITE_FG, "%16.8f", vec[3]);
    print(PRINT_LIME_FG, ")\n");
} /* verbose_vec4 */


/* Print ivec4_t */
extern void verbose_ivec4(
    ivec4_t vec // Vector
) {
    print(PRINT_LIME_FG, " (");
    print(PRINT_WHITE_FG, "%16i", vec[0]);
    print(PRINT_GREEN_FG, ", ");
    print(PRINT_WHITE_FG, "%16i", vec[1]);
    print(PRINT_GREEN_FG, ", ");
    print(PRINT_WHITE_FG, "%16i", vec[2]);
    print(PRINT_GREEN_FG, ", ");
    print(PRINT_WHITE_FG, "%16i", vec[3]);
    print(PRINT_LIME_FG, ")\n");
} /* verbose_ivec4 */


/* Vector array ------------------------------------------------------------- */


/* Print vec2_t array */
extern void verbose_vec2_array(
    size_t size, // Size of array
    vec2_t *vec  // Vector
) {
    for (size_t i = 0; i < size; i++) {
        if (i < 2 || size - 3 < i) {
            verbose_vec2(vec[i]);
        } else if (i == 2) {
            print(PRINT_GRAY_FG, " . . .\n");
        }
    }
} /* verbose_vec2_array */


/* Print ivec2_t array */
extern void verbose_ivec2_array(
    size_t size, // Size of array
    ivec2_t *vec // Vector
) {
    for (size_t i = 0; i < size; i++) {
        if (i < 2 || size - 3 < i) {
            verbose_ivec2(vec[i]);
        } else if (i == 2) {
            print(PRINT_GRAY_FG, " . . .\n");
        }
    }
} /* verbose_ivec2_array */


/* Print vec3_t array */
extern void verbose_vec3_array(
    size_t size, // Size of array
    vec3_t *vec  // Vector
) {
    for (size_t i = 0; i < size; i++) {
        if (i < 2 || size - 3 < i) {
            verbose_vec3(vec[i]);
        } else if (i == 2) {
            print(PRINT_GRAY_FG, " . . .\n");
        }
    }
} /* verbose_vec3_array */


/* Print ivec3_t array */
extern void verbose_ivec3_array(
    size_t size, // Size of array
    ivec3_t *vec // Vector
) {
    for (size_t i = 0; i < size; i++) {
        if (i < 2 || size - 3 < i) {
            verbose_ivec3(vec[i]);
        } else if (i == 2) {
            print(PRINT_GRAY_FG, " . . .\n");
        }
    }
} /* verbose_ivec3_array */


/* Print vec4_t array */
extern void verbose_vec4_array(
    size_t size, // Size of array
    vec4_t *vec  // Vector
) {
    for (size_t i = 0; i < size; i++) {
        if (i < 2 || size - 3 < i) {
            verbose_vec4(vec[i]);
        } else if (i == 2) {
            print(PRINT_GRAY_FG, " . . .\n");
        }
    }
} /* verbose_vec4_array */


/* Print ivec4_t array */
extern void verbose_ivec4_array(
    size_t size, // Size of array
    ivec4_t *vec // Vector
) {
    for (size_t i = 0; i < size; i++) {
        if (i < 2 || size - 3 < i) {
            verbose_ivec4(vec[i]);
        } else if (i == 2) {
            print(PRINT_GRAY_FG, " . . .\n");
        }
    }
} /* verbose_ivec4_array */


/* Matrix ------------------------------------------------------------------- */


/* Print mat3 */
extern void verbose_mat3(
    mat3_t mat // Matrix
) {
    print(PRINT_LIME_FG, " /");
    print(PRINT_WHITE_FG, "%16.8f", mat[0]);
    print(PRINT_GREEN_FG, ", ");
    print(PRINT_WHITE_FG, "%16.8f", mat[1]);
    print(PRINT_GREEN_FG, ", ");
    print(PRINT_WHITE_FG, "%16.8f", mat[2]);
    print(PRINT_LIME_FG, "\\\n");

    print(PRINT_LIME_FG, " |");
    print(PRINT_WHITE_FG, "%16.8f", mat[3]);
    print(PRINT_GREEN_FG, ", ");
    print(PRINT_WHITE_FG, "%16.8f", mat[4]);
    print(PRINT_GREEN_FG, ", ");
    print(PRINT_WHITE_FG, "%16.8f", mat[5]);
    print(PRINT_LIME_FG, "|\n");

    print(PRINT_LIME_FG, " \\");
    print(PRINT_WHITE_FG, "%16.8f", mat[6]);
    print(PRINT_GREEN_FG, ", ");
    print(PRINT_WHITE_FG, "%16.8f", mat[7]);
    print(PRINT_GREEN_FG, ", ");
    print(PRINT_WHITE_FG, "%16.8f", mat[8]);
    print(PRINT_LIME_FG, "/\n");
} /* verbose_mat3 */


/* Print mat4 */
extern void verbose_mat4(
    mat4_t mat // Matrix
) {
    print(PRINT_LIME_FG, " /");
    print(PRINT_WHITE_FG, "%16.8f", mat[0]);
    print(PRINT_GREEN_FG, ", ");
    print(PRINT_WHITE_FG, "%16.8f", mat[1]);
    print(PRINT_GREEN_FG, ", ");
    print(PRINT_WHITE_FG, "%16.8f", mat[2]);
    print(PRINT_GREEN_FG, ", ");
    print(PRINT_WHITE_FG, "%16.8f", mat[3]);
    print(PRINT_LIME_FG, "\\\n");

    for (char i = 1; i < 3; i++) {
        print(PRINT_LIME_FG, " |");
        print(PRINT_WHITE_FG, "%16.8f", mat[i * 4 + 0]);
        print(PRINT_GREEN_FG, ", ");
        print(PRINT_WHITE_FG, "%16.8f", mat[i * 4 + 1]);
        print(PRINT_GREEN_FG, ", ");
        print(PRINT_WHITE_FG, "%16.8f", mat[i * 4 + 2]);
        print(PRINT_GREEN_FG, ", ");
        print(PRINT_WHITE_FG, "%16.8f", mat[i * 4 + 3]);
        print(PRINT_LIME_FG, "|\n");
    }

    print(PRINT_LIME_FG, " \\");
    print(PRINT_WHITE_FG, "%16.8f", mat[12]);
    print(PRINT_GREEN_FG, ", ");
    print(PRINT_WHITE_FG, "%16.8f", mat[13]);
    print(PRINT_GREEN_FG, ", ");
    print(PRINT_WHITE_FG, "%16.8f", mat[14]);
    print(PRINT_GREEN_FG, ", ");
    print(PRINT_WHITE_FG, "%16.8f", mat[15]);
    print(PRINT_LIME_FG, "/\n");
} /* verbose_mat4 */


/* Primitives --------------------------------------------------------------- */


/* Print rgb_t */
extern void verbose_rgb(
    rgb_t color // Color
) {
    print(PRINT_LIME_FG, " R: ");
    print(PRINT_WHITE_FG, "%02x ", color.r);
    print(PRINT_GRAY_FG, "%03i", color.r);
    print(PRINT_GREEN_FG, ",");
    print(PRINT_LIME_FG, " G: ");
    print(PRINT_WHITE_FG, "%02x ", color.g);
    print(PRINT_GRAY_FG, "%03i", color.g);
    print(PRINT_GREEN_FG, ",");
    print(PRINT_LIME_FG, " B: ");
    print(PRINT_WHITE_FG, "%02x ", color.b);
    print(PRINT_GRAY_FG, "%03i\n", color.b);
} /* verbose_rgb */


/* Print vertex_t */
extern void verbose_vertex(
    vertex_t ver // Vertex
) {
    print(PRINT_GRAY_FG, "          position\n");
    verbose_vec3(ver.position);
    print(PRINT_GRAY_FG, "             light\n");
    verbose_vec3(ver.light);
    print(PRINT_GRAY_FG, "              half\n");
    verbose_vec3(ver.half);
} /* verbose_vertex */


/* Print vertex_t array */
extern void verbose_vertex_array(
    size_t size,  // Size of array
    vertex_t *ver // Array
) {
    for (size_t i = 0; i < size; i++) {
        if (i < 2 || size - 3 < i) {
            print(PRINT_BLUE_BG, " %llu \n", i + 1);
            verbose_vertex(ver[i]);
        } else if (i == 2) {
            print(PRINT_GRAY_FG, " . . .\n");
        }
    }
} /* verbose_vertex_array */


/* Mesh --------------------------------------------------------------------- */


/* Print mesh_t */
extern void verbose_mesh(
    mesh_t *mesh // Mesh
) {
    print(PRINT_CYAN_BG, " Mesh %s.obj \n", global_mesh);
    print(PRINT_AQUA_BG, " %llu vertices \n", mesh->size_vertices);
    verbose_vec3_array(mesh->size_vertices, mesh->vertices);
    print(PRINT_AQUA_BG, " %llu normals \n", mesh->size_normals);
    verbose_vec3_array(mesh->size_normals, mesh->normals);
    print(PRINT_AQUA_BG, " %llu indices \n", mesh->size_indices);
    print(PRINT_GRAY_FG, "            vertex            normal\n");
    verbose_ivec2_array(mesh->size_indices, mesh->indices);
} /* verbose_mesh */


/* Scene -------------------------------------------------------------------- */


/* Print scene_t */
extern void verbose_scene(
    scene_t *scene // Scene
) {
    print(PRINT_CYAN_BG, " Scene \n");

    // Model
    print(PRINT_AQUA_BG, " Model \n");
    print(PRINT_BLUE_BG, " Mesh %s.obj \n", global_mesh);
    print(PRINT_WHITE_FG, " %llu vertices, %llu normals, %llu triangles \n",
        scene->model.size_vertices,
        scene->model.size_normals,
        scene->model.size_indices / 3
    );
    print(PRINT_BLUE_BG, " Position \n");
    verbose_vec3(scene->model_position);
    print(PRINT_BLUE_BG, " Scale \n");
    verbose_vec3(scene->model_scale);
    print(PRINT_BLUE_BG, " Rotation \n");
    verbose_vec3(scene->model_rotation);

    // Camera
    print(PRINT_AQUA_BG, " Camera \n");
    print(PRINT_BLUE_BG, " Field of view \n");
    print(PRINT_WHITE_FG, "  %16.8f\n", scene->camera_fov_y);
    print(PRINT_BLUE_BG, " Clipping planes \n");
    print(PRINT_GRAY_FG, "              near               far\n");
    verbose_vec2(scene->camera_clip_planes);
    print(PRINT_BLUE_BG, " Configuration \n");
    print(PRINT_GRAY_FG,
        "         longitude          latitude          distance\n");
    verbose_vec3(scene->camera_configuration);

    // Light
    print(PRINT_AQUA_BG, " Light \n");
    print(PRINT_BLUE_BG, " Configuration \n");
    print(PRINT_GRAY_FG,
        "         longitude          latitude          distance\n");
    verbose_vec3(scene->light_configuration);
    print(PRINT_BLUE_BG, " Component constants \n");
    print(PRINT_GRAY_FG,
        "           ambient           diffuse          specular\n");
    verbose_vec3(scene->light_constant);
    print(PRINT_BLUE_BG, " Colors \n");
    print(PRINT_GRAY_FG, "   ambient\n");
    verbose_rgb(scene->light_ambient);
    print(PRINT_GRAY_FG, "   diffuse\n");
    verbose_rgb(scene->light_diffuse);
    print(PRINT_GRAY_FG, "   specular\n");
    verbose_rgb(scene->light_specular);
    print(PRINT_BLUE_BG, " Shininess \n");
    print(PRINT_WHITE_FG, " %16i\n", scene->light_shininess);
} /* verbose_scene */


/* -------------------------------------------------------------------------- */


#endif /* _DEBUG */


/* verbose.c */