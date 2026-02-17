/* CUDA C Software Rendering Pipeline ---------------------------------------
   Bachelor Thesis                                              Jakub Profota
   Summer Semester 2022/3                    OI - Computer Games and Graphics
   Supervisor Jiri Bittner               Czech Technical University in Prague
   math.cu ------------------------------------------------------------------ */


#include "math.cuh"

#include <math.h> // sinf, cosf


/* Vector ------------------------------------------------------------------- */


/* Subtract vectors */
extern inline void vec3_subtract(
    vec3_t a,  // Left vector
    vec3_t b,  // Right vector
    vec3_t vec // Target vector
) {
    vec[0] = a[0] - b[0];
    vec[1] = a[1] - b[1];
    vec[2] = a[2] - b[2];
} /* vec3_subtract */


/* Cross product */
extern inline void vec3_cross(
    vec3_t a,  // Vector
    vec3_t b,  // Vector
    vec3_t vec // Target vector
) {
    vec3_t c;
    c[0] = a[1] * b[2] - a[2] * b[1];
    c[1] = a[2] * b[0] - a[0] * b[2];
    c[2] = a[0] * b[1] - a[1] * b[0];
    vec[0] = c[0];
    vec[1] = c[1];
    vec[2] = c[2];
} /* vec3_cross */


/* Longitude latitude configuration to cartesian coordinates */
void vec3_position(
    vec3_t config, // Configuration
    vec3_t pos     // Target position
) {
    float s0, c0, s1, c1;
    sincosf(((90.0f - config[0]) * MATH_PI) / 180.0f, &s0, &c0);
    sincosf(((90.0f - config[1]) * MATH_PI) / 180.0f, &s1, &c1);
    pos[0] = config[2] * s1 * c0;
    pos[1] = config[2] * s1 * s0;
    pos[2] = config[2] * c1;
} /* vec3_position */


/* Matrix ------------------------------------------------------------------- */


/* 3x3 matrix to 4x4 matrix */
extern inline void mat3_to_mat4(
    mat3_t a,  // Matrix
    mat4_t mat // Target matrix
) {
    mat[0] = a[0];
    mat[1] = a[1];
    mat[2] = a[2];

    mat[4] = a[3];
    mat[5] = a[4];
    mat[6] = a[5];

    mat[8] = a[6];
    mat[9] = a[7];
    mat[10] = a[8];

    mat[15] = 1.0f;
} /* mat3_to_mat4 */


/* Copy matrix */
void mat4_copy(
    mat4_t a,  // Matrix
    mat4_t mat // Target matrix
) {
    mat[0] = a[0];
    mat[1] = a[1];
    mat[2] = a[2];
    mat[3] = a[3];
    mat[4] = a[4];
    mat[5] = a[5];
    mat[6] = a[6];
    mat[7] = a[7];
    mat[8] = a[8];
    mat[9] = a[9];
    mat[10] = a[10];
    mat[11] = a[11];
    mat[12] = a[12];
    mat[13] = a[13];
    mat[14] = a[14];
    mat[15] = a[15];
} /* mat4_copy */


/* Translate matrix */
extern inline void mat4_translate_xyz(
    vec3_t vec, // Vector with values
    mat4_t mat  // Target matrix
) {
    mat[3] = vec[0];
    mat[7] = vec[1];
    mat[11] = vec[2];
} /* mat4_translate_xyz */


/* Multiply matrix with matrix */
extern inline void mat4_multiply(
    mat4_t a,  // Left matrix
    mat4_t b,  // Right matrix
    mat4_t mat // Target matrix
) {
    float a00 = a[0], a01 = a[1], a02 = a[2], a03 = a[3];
    float a10 = a[4], a11 = a[5], a12 = a[6], a13 = a[7];
    float a20 = a[8], a21 = a[9], a22 = a[10], a23 = a[11];
    float a30 = a[12], a31 = a[13], a32 = a[14], a33 = a[15];

    float b00 = b[0], b01 = b[1], b02 = b[2], b03 = b[3];
    float b10 = b[4], b11 = b[5], b12 = b[6], b13 = b[7];
    float b20 = b[8], b21 = b[9], b22 = b[10], b23 = b[11];
    float b30 = b[12], b31 = b[13], b32 = b[14], b33 = b[15];

    mat[0] = a00 * b00 + a01 * b10 + a02 * b20 + a03 * b30;
    mat[1] = a00 * b01 + a01 * b11 + a02 * b21 + a03 * b31;
    mat[2] = a00 * b02 + a01 * b12 + a02 * b22 + a03 * b32;
    mat[3] = a00 * b03 + a01 * b13 + a02 * b23 + a03 * b33;

    mat[4] = a10 * b00 + a11 * b10 + a12 * b20 + a13 * b30;
    mat[5] = a10 * b01 + a11 * b11 + a12 * b21 + a13 * b31;
    mat[6] = a10 * b02 + a11 * b12 + a12 * b22 + a13 * b32;
    mat[7] = a10 * b03 + a11 * b13 + a12 * b23 + a13 * b33;

    mat[8] = a20 * b00 + a21 * b10 + a22 * b20 + a23 * b30;
    mat[9] = a20 * b01 + a21 * b11 + a22 * b21 + a23 * b31;
    mat[10] = a20 * b02 + a21 * b12 + a22 * b22 + a23 * b32;
    mat[11] = a20 * b03 + a21 * b13 + a22 * b23 + a23 * b33;

    mat[12] = a30 * b00 + a31 * b10 + a32 * b20 + a33 * b30;
    mat[13] = a30 * b01 + a31 * b11 + a32 * b21 + a33 * b31;
    mat[14] = a30 * b02 + a31 * b12 + a32 * b22 + a33 * b32;
    mat[15] = a30 * b03 + a31 * b13 + a32 * b23 + a33 * b33;
} /* mat4_multiply */


/* Transpose matrix */
void mat4_transpose(
    mat4_t mat // Target matrix
) {
    float m01 = mat[1], m02 = mat[2], m03 = mat[3];
    float m10 = mat[4], m12 = mat[6], m13 = mat[7];
    float m20 = mat[8], m21 = mat[9], m23 = mat[11];
    float m30 = mat[12], m31 = mat[13], m32 = mat[14];

    mat[1] = m10;
    mat[2] = m20;
    mat[3] = m30;
    mat[4] = m01;
    mat[6] = m21;
    mat[7] = m31;
    mat[8] = m02;
    mat[9] = m12;
    mat[11] = m32;
    mat[12] = m03;
    mat[13] = m13;
    mat[14] = m23;
} /* mat4_transpose */


/* Invert matrix */
void mat4_inverse(
    mat4_t mat // Target matrix
) {
    mat4_t inv;
    inv[0] = mat[5] * mat[10] * mat[15] - mat[5] * mat[11] * mat[14] -
             mat[9] * mat[6]  * mat[15] + mat[9] * mat[7]  * mat[14] +
             mat[13] * mat[6]  * mat[11] - mat[13] * mat[7]  * mat[10];

    inv[4] = -mat[4] * mat[10] * mat[15] + mat[4] * mat[11] * mat[14] +
              mat[8] * mat[6]  * mat[15] - mat[8] * mat[7]  * mat[14] -
              mat[12] * mat[6]  * mat[11] + mat[12] * mat[7]  * mat[10];

    inv[8] = mat[4] * mat[9] * mat[15] - mat[4] * mat[11] * mat[13] -
             mat[8] * mat[5] * mat[15] + mat[8] * mat[7] * mat[13] +
             mat[12] * mat[5] * mat[11] - mat[12] * mat[7] * mat[9];

    inv[12] = -mat[4] * mat[9] * mat[14] + mat[4] * mat[10] * mat[13] +
               mat[8] * mat[5] * mat[14] - mat[8] * mat[6] * mat[13] -
               mat[12] * mat[5] * mat[10] + mat[12] * mat[6] * mat[9];

    inv[1] = -mat[1] * mat[10] * mat[15] + mat[1] * mat[11] * mat[14] +
              mat[9] * mat[2] * mat[15] - mat[9] * mat[3] * mat[14] -
              mat[13] * mat[2] * mat[11] + mat[13] * mat[3] * mat[10];

    inv[5] = mat[0] * mat[10] * mat[15] - mat[0] * mat[11] * mat[14] -
             mat[8] * mat[2] * mat[15] + mat[8] * mat[3] * mat[14] +
             mat[12] * mat[2] * mat[11] - mat[12] * mat[3] * mat[10];

    inv[9] = -mat[0] * mat[9] * mat[15] + mat[0] * mat[11] * mat[13] +
              mat[8] * mat[1] * mat[15] - mat[8] * mat[3] * mat[13] -
              mat[12] * mat[1] * mat[11] + mat[12] * mat[3] * mat[9];

    inv[13] = mat[0] * mat[9] * mat[14] - mat[0] * mat[10] * mat[13] -
              mat[8] * mat[1] * mat[14] + mat[8] * mat[2] * mat[13] +
              mat[12] * mat[1] * mat[10] - mat[12] * mat[2] * mat[9];

    inv[2] = mat[1] * mat[6] * mat[15] - mat[1] * mat[7] * mat[14] -
             mat[5] * mat[2] * mat[15] + mat[5] * mat[3] * mat[14] +
             mat[13] * mat[2] * mat[7] - mat[13] * mat[3] * mat[6];

    inv[6] = -mat[0] * mat[6] * mat[15] + mat[0] * mat[7] * mat[14] +
              mat[4] * mat[2] * mat[15] - mat[4] * mat[3] * mat[14] -
              mat[12] * mat[2] * mat[7] + mat[12] * mat[3] * mat[6];

    inv[10] = mat[0] * mat[5] * mat[15] - mat[0] * mat[7] * mat[13] -
              mat[4] * mat[1] * mat[15] + mat[4] * mat[3] * mat[13] +
              mat[12] * mat[1] * mat[7] - mat[12] * mat[3] * mat[5];

    inv[14] = -mat[0] * mat[5] * mat[14] + mat[0] * mat[6] * mat[13] +
               mat[4] * mat[1] * mat[14] - mat[4] * mat[2] * mat[13] -
               mat[12] * mat[1] * mat[6] + mat[12] * mat[2] * mat[5];

    inv[3] = -mat[1] * mat[6] * mat[11] + mat[1] * mat[7] * mat[10] +
              mat[5] * mat[2] * mat[11] - mat[5] * mat[3] * mat[10] -
              mat[9] * mat[2] * mat[7] + mat[9] * mat[3] * mat[6];

    inv[7] = mat[0] * mat[6] * mat[11] - mat[0] * mat[7] * mat[10] -
             mat[4] * mat[2] * mat[11] + mat[4] * mat[3] * mat[10] +
             mat[8] * mat[2] * mat[7] - mat[8] * mat[3] * mat[6];

    inv[11] = -mat[0] * mat[5] * mat[11] + mat[0] * mat[7] * mat[9] +
               mat[4] * mat[1] * mat[11] - mat[4] * mat[3] * mat[9] -
               mat[8] * mat[1] * mat[7] + mat[8] * mat[3] * mat[5];

    inv[15] = mat[0] * mat[5] * mat[10] - mat[0] * mat[6] * mat[9] -
              mat[4] * mat[1] * mat[10] + mat[4] * mat[2] * mat[9] +
              mat[8] * mat[1] * mat[6] - mat[8] * mat[2] * mat[5];

    float det = mat[0] * inv[0] + mat[1] * inv[4] +
                mat[2] * inv[8] + mat[3] * inv[12];
    det = 1.0f / det;

    mat[0] = inv[0] * det;
    mat[1] = inv[1] * det;
    mat[2] = inv[2] * det;
    mat[3] = inv[3] * det;
    mat[4] = inv[4] * det;
    mat[5] = inv[5] * det;
    mat[6] = inv[6] * det;
    mat[7] = inv[7] * det;
    mat[8] = inv[8] * det;
    mat[9] = inv[9] * det;
    mat[10] = inv[10] * det;
    mat[11] = inv[11] * det;
    mat[12] = inv[12] * det;
    mat[13] = inv[13] * det;
    mat[14] = inv[14] * det;
    mat[15] = inv[15] * det;
} /* mat4_inverse */


/* View matrix */
extern inline void mat4_lookat(
    vec3_t eye,    // Camera position
    vec3_t center, // View target
    vec3_t up,     // Up vector
    mat4_t mat     // Target matrix
) {
    vec3_t l = { 0 };
    vec3_subtract(center, eye, l);
    vec3_normalize(l);

    vec3_t p = { 0 };
    vec3_cross(l, up, p);
    vec3_normalize(p);

    vec3_t h = { 0 };
    vec3_cross(p, l, h);

    mat[0] = p[0];
    mat[1] = p[1];
    mat[2] = p[2];
    mat[3] = -vec3_dot(p, eye);

    mat[4] = h[0];
    mat[5] = h[1];
    mat[6] = h[2];
    mat[7] = -vec3_dot(h, eye);

    mat[8] = -l[0];
    mat[9] = -l[1];
    mat[10] = -l[2];
    mat[11] = vec3_dot(l, eye);

    mat[12] = 0.0f;
    mat[13] = 0.0f;
    mat[14] = 0.0f;
    mat[15] = 1.0f;
} /* mat4_lookat */


/* Perspective matrix */
extern inline void mat4_perspective(
    float fov_y,        // Vertical field of view
    float aspect_ratio, // Frame aspect ratio
    vec2_t clip_planes, // Near and far clipping plane
    mat4_t mat          // Target matrix
) {
    fov_y = (fov_y * MATH_PI) / 180.0f;
    float h = 1.0f / tanf(fov_y * 0.5f);
    float div = 1.0f / (clip_planes[0] - clip_planes[1]);

    mat[0] = h / aspect_ratio;
    mat[5] = h;
    mat[10] = (clip_planes[0] + clip_planes[1]) * div;
    mat[11] = 2.0f * clip_planes[0] * clip_planes[1] * div;
    mat[14] = -1.0f;
} /* mat4_perspective */


/* -------------------------------------------------------------------------- */


/* math.cu */