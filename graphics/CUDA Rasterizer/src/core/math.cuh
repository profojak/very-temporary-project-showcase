/* CUDA C Software Rendering Pipeline ---------------------------------------
   Bachelor Thesis                                              Jakub Profota
   Summer Semester 2022/3                    OI - Computer Games and Graphics
   Supervisor Jiri Bittner               Czech Technical University in Prague
   math.cuh ----------------------------------------------------------------- */


#ifndef MATH_H
#define MATH_H


#include "define.h"


// Vector
typedef float vec2_t[2];
typedef int ivec2_t[2];
typedef float vec3_t[3];
typedef int ivec3_t[3];
typedef float vec4_t[4];
typedef int ivec4_t[4];

// Matrix
typedef float mat3_t[9];
typedef float mat4_t[16];

// Primitives
typedef struct rgb_t {
    unsigned char r;
    unsigned char g;
    unsigned char b;
} rgb_t;

typedef struct vertex_t {
    vec3_t position;
    vec3_t light;
    vec3_t half;
} vertex_t;

typedef struct triangle_t {
    vertex_t vertices[3];
    vec3_t normals[3];
    char discard[3];
} triangle_t;

typedef struct fragment_t {
    vec3_t normal;
    vec3_t light;
    vec3_t half;
    float depth;
} fragment_t;


void vec3_subtract(vec3_t, vec3_t, vec3_t);
void vec3_cross(vec3_t, vec3_t, vec3_t);
void vec3_position(vec3_t, vec3_t);

void mat3_to_mat4(mat3_t, mat4_t);
void mat4_copy(mat4_t, mat4_t);
void mat4_translate_xyz(vec3_t, mat4_t);
void mat4_multiply(mat4_t, mat4_t, mat4_t);
void mat4_transpose(mat4_t);
void mat4_inverse(mat4_t);
void mat4_lookat(vec3_t, vec3_t, vec3_t, mat4_t);
void mat4_perspective(float, float, vec2_t, mat4_t);


/* Float -------------------------------------------------------------------- */


/* Minimum */
__device__ extern inline float minimum(
    float a, // Float
    float b  // Float
) {
    return a < b ? a : b;
} /* min */


/* Maximum */
__device__ extern inline float maximum(
    float a, // Float
    float b  // Float
) {
    return a > b ? a : b;
} /* max */


/* Vector ------------------------------------------------------------------- */


/* Dot product */
__host__ __device__ extern inline float vec3_dot(
    vec3_t a, // Vector
    vec3_t b  // Vector
) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
} /* vec3_dot */


/* Normalize vector */
__host__ __device__ extern inline void vec3_normalize(
    vec3_t vec // Target vector
) {
    float norm = sqrtf(vec3_dot(vec, vec));

    if (norm == 0.0f) {
        vec[0] = 0.0f;
        vec[1] = 0.0f;
        vec[2] = 0.0f;
    } else {
        vec[0] /= norm;
        vec[1] /= norm;
        vec[2] /= norm;
    }
} /* vec3_normalize */


/* Multiply vector with matrix */
__host__ __device__ extern inline void vec3_multiply_mat(
    mat3_t m,  // Matrix
    vec3_t v,  // Vector
    vec3_t vec // Target vector
) {
    vec3_t t;
    t[0] = m[0] * v[0] + m[1] * v[1] + m[2] * v[2];
    t[1] = m[3] * v[0] + m[4] * v[1] + m[5] * v[2];
    t[2] = m[6] * v[0] + m[7] * v[1] + m[8] * v[2];
    vec[0] = t[0];
    vec[1] = t[1];
    vec[2] = t[2];
} /* vec3_multiply_mat */


/* Subtract vectors */
__host__ __device__ extern inline void vec4_subtract(
    vec4_t a,  // Left vector
    vec4_t b,  // Right vector
    vec4_t vec // Target vector
) {
    vec[0] = a[0] - b[0];
    vec[1] = a[1] - b[1];
    vec[2] = a[2] - b[2];
    vec[3] = a[3] - b[3];
} /* vec4_subtract */


/* Multiply vector with matrix */
__host__ __device__ extern inline void vec4_multiply_mat(
    mat4_t m,  // Matrix
    vec4_t v,  // Vector
    vec4_t vec // Target vector
) {
    vec4_t t;
    t[0] = m[0] * v[0] + m[1] * v[1] + m[2] * v[2] + m[3] * v[3];
    t[1] = m[4] * v[0] + m[5] * v[1] + m[6] * v[2] + m[7] * v[3];
    t[2] = m[8] * v[0] + m[9] * v[1] + m[10] * v[2] + m[11] * v[3];
    t[3] = m[12] * v[0] + m[13] * v[1] + m[14] * v[2] + m[15] * v[3];
    vec[0] = t[0];
    vec[1] = t[1];
    vec[2] = t[2];
    vec[3] = t[3];
} /* vec4_multiply_mat */


/* Matrix ------------------------------------------------------------------- */


/* Identity matrix */
extern inline void mat3_identity(
    mat3_t mat // Target matrix
) {
    memset(mat, 0, sizeof(mat3_t));
    mat[0] = 1.0f;
    mat[4] = 1.0f;
    mat[8] = 1.0f;
} /* mat3_identity */


/* Multiply matrix with matrix */
extern inline void mat3_mulitply(
    mat3_t a,  // Left matrix
    mat3_t b,  // Right matrix
    mat3_t mat // Target matrix
) {
    float a00 = a[0], a01 = a[1], a02 = a[2];
    float a10 = a[3], a11 = a[4], a12 = a[5];
    float a20 = a[6], a21 = a[7], a22 = a[8];

    float b00 = b[0], b01 = b[1], b02 = b[2];
    float b10 = b[3], b11 = b[4], b12 = b[5];
    float b20 = b[6], b21 = b[7], b22 = b[8];

    mat[0] = a00 * b00 + a01 * b10 + a02 * b20;
    mat[1] = a00 * b01 + a01 * b11 + a02 * b21;
    mat[2] = a00 * b02 + a01 * b12 + a02 * b22;

    mat[3] = a10 * b00 + a11 * b10 + a12 * b20;
    mat[4] = a10 * b01 + a11 * b11 + a12 * b21;
    mat[5] = a10 * b02 + a11 * b12 + a12 * b22;

    mat[6] = a20 * b00 + a21 * b10 + a22 * b20;
    mat[7] = a20 * b01 + a21 * b11 + a22 * b21;
    mat[8] = a20 * b02 + a21 * b12 + a22 * b22;
} /* mat3_multiply */


/* Scale matrix */
extern inline void mat3_scale_xyz(
    vec3_t vec, // Vector with values
    mat3_t mat  // Target matrix
) {
    mat[0] *= vec[0];
    mat[4] *= vec[1];
    mat[8] *= vec[2];
} /* mat3_scale_xyz */


/* Rotate around x axis */
extern inline void mat3_rotate_x(
    float ang, // Angle
    mat3_t mat // Target matrix
) {
    ang = (ang * MATH_PI) / 180.0f;
    float s, c;
    sincosf(ang, &s, &c);

    mat3_t rot = { 0 };
    rot[0] = 1.0f;
    rot[4] = c;
    rot[5] = -s;
    rot[7] = s;
    rot[8] = c;
    mat3_mulitply(rot, mat, mat);
} /* mat3_rotate_x */


/* Rotate around y axis */
extern inline void mat3_rotate_y(
    float ang, // Angle
    mat3_t mat // Target matrix
) {
    ang = (ang * MATH_PI) / 180.0f;
    float s, c;
    sincosf(ang, &s, &c);

    mat3_t rot = { 0 };
    rot[0] = c;
    rot[2] = s;
    rot[4] = 1.0f;
    rot[6] = -s;
    rot[8] = c;
    mat3_mulitply(rot, mat, mat);
} /* mat3_rotate_y */


/* Rotate around z axis */
extern inline void mat3_rotate_z(
    float ang, // Angle
    mat3_t mat // Target matrix
) {
    ang = (ang * MATH_PI) / 180.0f;
    float s, c;
    sincosf(ang, &s, &c);

    mat3_t rot = { 0 };
    rot[0] = c;
    rot[1] = -s;
    rot[3] = s;
    rot[4] = c;
    rot[8] = 1.0f;
    mat3_mulitply(rot, mat, mat);
} /* mat3_rotate_z */


/* Rotate matrix */
extern inline void mat3_rotate_xyz(
    vec3_t vec, // Vector with values
    mat3_t mat  // Target matrix
) {
    mat3_rotate_x(vec[0], mat);
    mat3_rotate_y(vec[1], mat);
    mat3_rotate_z(vec[2], mat);
} /* mat3_rotate_xyz */


/* Triangle ----------------------------------------------------------------- */


/* Get 2D axis aligned bounding box */
__device__ extern inline void triangle_aabb_2d(
    triangle_t triangle, // Triangle
    vec2_t min,          // Minimum
    vec2_t max           // Maximum
) {
    float v0x = triangle.vertices[0].position[0];
    float v0y = triangle.vertices[0].position[1];
    float v1x = triangle.vertices[1].position[0];
    float v1y = triangle.vertices[1].position[1];
    float v2x = triangle.vertices[2].position[0];
    float v2y = triangle.vertices[2].position[1];

    min[0] = minimum(minimum(v0x, v1x), v2x);
    min[1] = minimum(minimum(v0y, v1y), v2y);
    max[0] = maximum(maximum(v0x, v1x), v2x);
    max[1] = maximum(maximum(v0y, v1y), v2y);
} /* triangle_aabb_2d */


/* Get 3D axis aligned bounding box */
__device__ extern inline void triangle_aabb_3d(
    triangle_t triangle, // Triangle
    vec3_t min,          // Minimum
    vec3_t max           // Maximum
) {
    float v0x = triangle.vertices[0].position[0];
    float v0y = triangle.vertices[0].position[1];
    float v0z = triangle.vertices[0].position[2];
    float v1x = triangle.vertices[1].position[0];
    float v1y = triangle.vertices[1].position[1];
    float v1z = triangle.vertices[1].position[2];
    float v2x = triangle.vertices[2].position[0];
    float v2y = triangle.vertices[2].position[1];
    float v2z = triangle.vertices[2].position[2];

    min[0] = minimum(minimum(v0x, v1x), v2x);
    min[1] = minimum(minimum(v0y, v1y), v2y);
    min[2] = minimum(minimum(v0z, v1z), v2z);
    max[0] = maximum(maximum(v0x, v1x), v2x);
    max[1] = maximum(maximum(v0y, v1y), v2y);
    max[2] = maximum(maximum(v0z, v1z), v2z);
} /* triangle_aabb_3d */


/* Primitives --------------------------------------------------------------- */


/* Cartesian coordinates to barycentric coordinates of specified triangle */
//__device__ extern inline void cartesian_to_barycentric(
__device__ extern inline int cartesian_to_barycentric(
    vec3_t a,      // Triangle vertex
    vec3_t b,      // Triangle vertex
    vec3_t c,      // Triangle vertex
    vec2_t p,      // Point
    vec3_t *coords // Target barycentric coordinates
) {
    vec3_t v0, v1, v2;
    v0[0] = b[0] - a[0];
    v0[1] = b[1] - a[1];
    v0[2] = 0.0f;
    v1[0] = c[0] - a[0];
    v1[1] = c[1] - a[1];
    v1[2] = 0.0f;
    v2[0] = p[0] - a[0];
    v2[1] = p[1] - a[1];
    v2[2] = 0.0f;

    float d00 = vec3_dot(v0, v0);
    float d01 = vec3_dot(v0, v1);
    float d11 = vec3_dot(v1, v1);
    float d20 = vec3_dot(v2, v0);
    float d21 = vec3_dot(v2, v1);
    float det = d00 * d11 - d01 * d01;

    if (det > 0.0000001f) {
        det = 1.0f / det;
        (*coords)[1] = (d11 * d20 - d01 * d21) * det;
        (*coords)[2] = (d00 * d21 - d01 * d20) * det;
        (*coords)[0] = 1.0f - (*coords)[1] - (*coords)[2];
        return 1;
    } else {
        return 0;
    }
} /* cartesian_to_barycentric */


/* -------------------------------------------------------------------------- */


#endif /* MATH_H */


/* math.cuh */