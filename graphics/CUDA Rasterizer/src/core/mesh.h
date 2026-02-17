/* CUDA C Software Rendering Pipeline ---------------------------------------
   Bachelor Thesis                                              Jakub Profota
   Summer Semester 2022/3                    OI - Computer Games and Graphics
   Supervisor Jiri Bittner               Czech Technical University in Prague
   mesh.h ------------------------------------------------------------------- */


#ifndef MESH_H
#define MESH_H


#include "math.cuh"


typedef struct mesh_t {
    size_t size_vertices; // Size of vertex buffer
    size_t size_normals;  // Size of normal buffer
    size_t size_indices;  // Size of index buffer
    vec3_t *vertices;     // Vertex buffer
    vec3_t *normals;      // Normal buffer
    ivec2_t *indices;     // Index buffer: vertex, normal
} mesh_t;


int mesh_load(mesh_t *);
int mesh_unload(mesh_t *);


/* -------------------------------------------------------------------------- */


#endif /* MESH_H */


/* mesh.h */