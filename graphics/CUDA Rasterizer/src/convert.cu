/* CUDA C Software Rendering Pipeline ---------------------------------------
   Bachelor Thesis                                              Jakub Profota
   Summer Semester 2022/3                    OI - Computer Games and Graphics
   Supervisor Jiri Bittner               Czech Technical University in Prague
   convert.c ---------------------------------------------------------------- */


#include "core/math.cuh"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>


/* Entry -------------------------------------------------------------------- */


/* Entry point of .obj converter */
int main(int argc, char *argv[]) {
    if (argc != 9) {
        printf("Run converter like this:\n");
        printf("    .\\convert.exe <name> <scale> "
            "<rotate x> <rotate y> <rotate z> "
            "<translate x> <translate y> <translate z>\n");
        return EXIT_FAILURE;
    }
    char *name = argv[1];
    printf("Converting \"%s\" ...\n", name);

    float scale_factor = atof(argv[2]);
    float rotate_x = atof(argv[3]);
    float rotate_y = atof(argv[4]);
    float rotate_z = atof(argv[5]);
    float translate_x = atof(argv[6]);
    float translate_y = atof(argv[7]);
    float translate_z = atof(argv[8]);

    vec3_t scale_vec;
    scale_vec[0] = scale_factor;
    scale_vec[1] = scale_factor;
    scale_vec[2] = scale_factor;

    vec3_t rotate_vec;
    rotate_vec[0] = rotate_x;
    rotate_vec[1] = rotate_y;
    rotate_vec[2] = rotate_z;

    mat3_t scale_mat;
    mat3_identity(scale_mat);
    mat3_scale_xyz(scale_vec, scale_mat);
    mat3_t rotate_mat;
    mat3_identity(rotate_mat);
    mat3_rotate_xyz(rotate_vec, rotate_mat);

    size_t size_vertices = 0;
    size_t size_normals = 0;
    size_t size_triangles = 0;

    // Open object file
    char line[1024];
    char in_path[1024];
    char out_path[1024];
    memset(line, 0, 1024);
    memset(in_path, 0, 1024);
    memset(out_path, 0, 1024);

    strcat(in_path, "obj\\");
    strcat(in_path, name);
    strcat(in_path, ".obj");
    strcat(out_path, "obj\\");
    strcat(out_path, name);
    strcat(out_path, "_converted.obj");

    FILE *in = fopen(in_path, "r");
    FILE *out = fopen(out_path, "w");
    if (in == 0) {
        printf("Failed to open object file!\n");
        printf("Object file \"%s.obj\" may not exist\n", name);
        printf("Please run converter from root directory\n");
        return EXIT_FAILURE;
    }

    while (fgets(line, 1023, in) != 0) {

        // Vertex
        if (line[0] == 'v' && line[1] == ' ') {
            vec3_t v;
            sscanf(line, "v %f %f %f", &v[0], &v[1], &v[2]);
            vec3_multiply_mat(scale_mat, v, v);
            vec3_multiply_mat(rotate_mat, v, v);
            v[0] += translate_x;
            v[1] += translate_y;
            v[2] += translate_z;
            fprintf(out, "v %f %f %f\n", v[0], v[1], v[2]);
            size_vertices++;

        // Vertex normal
        } else if (line[0] == 'v' && line[1] == 'n') {
            vec3_t v;
            sscanf(line, "vn %f %f %f", &v[0], &v[1], &v[2]);
            vec3_multiply_mat(rotate_mat, v, v);
            fprintf(out, "vn %f %f %f\n", v[0], v[1], v[2]);
            size_normals++;

        // Face
        } else if (line[0] == 'f' && line[1] == ' ') {
            ivec3_t v0, v1, v2;
            if (sscanf(line, "f %i/%i/%i %i/%i/%i %i/%i/%i",
                &v0[0], &v0[1], &v0[2],
                &v1[0], &v1[1], &v1[2],
                &v2[0], &v2[1], &v2[2]
            ) == 9) {
                fprintf(out, "f %i//%i %i//%i %i//%i\n",
                    v0[0], v0[2], v1[0], v1[2], v2[0], v2[2]);
            } else {
                sscanf(line, "f %i//%i %i//%i %i//%i",
                    &v0[0], &v0[1], &v1[0], &v1[1], &v2[0], &v2[1]);
                fprintf(out, "f %i//%i %i//%i %i//%i\n",
                    v0[0], v0[1], v1[0], v1[1], v2[0], v2[1]);
            }
            size_triangles++;
        }
    }

    printf("%llu vertices, %llu normals, %llu triangles\n",
        size_vertices, size_normals, size_triangles);
    printf("Converted to \"%s_convert.obj\"", name);

    fclose(in);
    fclose(out);
    in = 0;
    out = 0;

    return EXIT_SUCCESS;
} /* main */


/* -------------------------------------------------------------------------- */


/* convert.c */