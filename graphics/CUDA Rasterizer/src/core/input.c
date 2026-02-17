/* CUDA C Software Rendering Pipeline ---------------------------------------
   Bachelor Thesis                                              Jakub Profota
   Summer Semester 2022/3                    OI - Computer Games and Graphics
   Supervisor Jiri Bittner               Czech Technical University in Prague
   input.c ------------------------------------------------------------------ */


#include "input.h"

#include "define.h"
#include "scene.h"
#include "../platform/print.h"
#include "../platform/window.h"


#define INPUT_MODEL_POSITION_DELTA 0.025f
#define INPUT_MODEL_SCALE_DELTA 0.1f
#define INPUT_MODEL_ROTATION_DELTA 2.5f
#define INPUT_CAM_FOV_DELTA 1.0f
#define INPUT_CAM_CLIP_NEAR_DELTA 2.0f
#define INPUT_CAM_CLIP_FAR_DELTA 1.0f
#define INPUT_CAM_LONGITUDE_DELTA_1 0.1f
#define INPUT_CAM_LONGITUDE_DELTA_2 0.5f
#define INPUT_CAM_LONGITUDE_DELTA_3 2.5f
#define INPUT_CAM_LATITUDE_DELTA_1 0.1f
#define INPUT_CAM_LATITUDE_DELTA_2 0.5f
#define INPUT_CAM_LATITUDE_DELTA_3 2.5f
#define INPUT_CAM_DISTANCE_DELTA_1 0.999f
#define INPUT_CAM_DISTANCE_DELTA_2 0.99f
#define INPUT_CAM_DISTANCE_DELTA_3 0.965f
#define INPUT_LIGHT_LONGITUDE_DELTA 2.5f
#define INPUT_LIGHT_LATITUDE_DELTA 2.5f
#define INPUT_LIGHT_DISTANCE_DELTA 0.1f
#define INPUT_LIGHT_SHININESS_DELTA 1


enum key_e {
    KEY_BACK = 0x08,    //    Back: verbose output
    KEY_RETURN = 0x0D,  //   Enter: idle benchmark
                    //    Sh Enter: 360 degrees benchmark
                    // Sh Ct Enter: zoom benchmark
    KEY_SHIFT = 0x10,   //   Shift: modifier key
    KEY_CONTROL = 0x11, // Control: modifier key
    KEY_ESCAPE = 0x1B,  //  Escape: quit
    KEY_1 = 0x31,       //       1: draw triangles
                        //    Sh 1: draw lines
                        // Sh Ct 1: draw points
    KEY_2 = 0x32,       //       2: smooth shading
                        //    Sh 2: flat shading
    KEY_3 = 0x33,       //       3: ambient component only
                        //    Sh 3: diffuse component only
                        // Sh Ct 3: specular component only
    KEY_4 = 0x34,       //       4: Blinn-Phong illumination
                        //    Sh 4: camera reflector illumination
    KEY_5 = 0x35,       //       5: normal vectors
                        //    Sh 5: depth buffer
    KEY_A = 0x41,       //       A: rotate camera left
                        //    Sh A: rotate camera left slower
                        // Sh Ct A: rotate camera left slowest
    KEY_B = 0x42,       //       B: rotate model along y axis
                        //    Sh B: rotate model along -y axis
                        // Sh Ct B: reset model y rotation
    KEY_C = 0x43,       //       C: move far clipping plane closer
                        //    Sh C: move far clipping plane farther
                        // Sh Ct C: reset far clipping plane
    KEY_D = 0x44,       //       D: rotate camera right
                        //    Sh D: rotate camera right slower
                        // Sh Ct D: rotate camera right slowest
    KEY_E = 0x45,       //       E: rotate camera up
                        //    Sh E: rotate camera up slower
                        // Sh Ct E: rotate camera up slowest
    KEY_F = 0x46,       //       F: scale model along x axis
                        //    Sh F: scale model along -x axis
                        // Sh Ct F: reset model x scale
    KEY_G = 0x47,       //       G: scale model along y axis
                        //    Sh G: scale model along -y axis
                        // Sh Ct G: reset model y scale
    KEY_H = 0x48,       //       H: scale model along z axis
                        //    Sh H: scale model along -z axis
                        // Sh Ct H: reset model z scale
    KEY_I = 0x49,       //       I: move light along y axis
                        //    Sh I: move light along -y axis
                        // Sh Ct I: reset light y position
    KEY_M = 0x4D,       //       M: toggle backface culling
                        //    Sh M: toggle view frustum clipping
    KEY_N = 0x4E,       //       E: rotate model along z axis
                        //    Sh E: rotate model along -z axis
                        // Sh Ct E: reset model z rotation
    KEY_O = 0x4F,       //       O: move light along z axis
                        //    Sh O: move light along -z axis
                        // Sh Ct O: reset light z position
    KEY_P = 0x50,       //       P: increase light shininess
                        //    Sh P: decrease light shininess
    KEY_Q = 0x51,       //       Q: rotate camera down
                        //    Sh Q: rotate camera down slower
                        // Sh Ct Q: rotate camera down slowest
    KEY_R = 0x52,       //       R: move model along x axis
                        //    Sh R: move model along -x axis
                        // Sh Ct R: reset model x position
    KEY_S = 0x53,       //       W: move camera farther
                        //    Sh W: move camera farther slower
                        // Sh Ct W: move camera farther slowest
    KEY_T = 0x54,       //       T: move model along y axis
                        //    Sh T: move model along -y axis
                        // Sh Ct T: reset model y position
    KEY_U = 0x55,       //       U: move light along x axis
                        //    Sh U: move light along -x axis
                        // Sh Ct U: reset light x position
    KEY_V = 0x56,       //       V: rotate model along x axis
                        //    Sh V: rotate model along -x axis
                        // Sh Ct V: reset model x rotation
    KEY_W = 0x57,       //       W: move camera closer
                        //    Sh W: move camera closer slower
                        // Sh Ct W: move camera closer slowest
    KEY_X = 0x58,       //       X: move near clipping plane closer
                        //    Sh X: move near clipping plane farther
                        // Sh Ct X: reset near clipping plane
    KEY_Y = 0x59,       //       Y: move model along z axis
                        //    Sh Y: move model along -z axis
                        // Sh Ct Y: reset model z position
    KEY_Z = 0x5A,       //       Z: narrower field of view
                        //    Sh Z: wider field of view
                        // Sh Ct Z: reset field of view
};


static unsigned char modifier_shift = 0;
static unsigned char modifier_control = 0;


#ifdef _DEBUG
int global_primitive = PRIMITIVE_TRIANGLE;
int global_shading = SHADING_SMOOTH;
int global_color = COLOR_BLINN_PHONG;
int global_culling = CULLING_BACKFACE | CULLING_FRUSTUM;
#endif /* _DEBUG */


/* Input -------------------------------------------------------------------- */


/* Process key press input */
extern void input_down(
    unsigned long long key, // Key
    int already_pressed     // If key was already pressed and is held down
) {
    if (global_benchmark != 0) {
        return;
    }

    switch (key) {

    case KEY_BACK: {
        if (!already_pressed) {
            global_verbose = VERBOSE_QUEUED;
        }
    } break;

    case KEY_RETURN: {
        if (!already_pressed) {
            if (modifier_control && modifier_shift) {
                global_benchmark = BENCHMARK_ZOOM;
            } else if (modifier_shift) {
                global_benchmark = BENCHMARK_360;
            } else {
                global_benchmark = BENCHMARK_IDLE;
            }
        }
    } break;

    case KEY_SHIFT: {
        if (!already_pressed) {
            modifier_shift = 1;
        }
    } break;

    case KEY_CONTROL: {
        if (!already_pressed) {
            modifier_control = 1;
        }
    } break;

    case KEY_ESCAPE: {
        window_set_status(WINDOW_SHUTTING_DOWN);
    } break;

    #ifdef _DEBUG
    case KEY_1: {
        if (modifier_control && modifier_shift) {
            global_primitive = PRIMITIVE_POINT;
            if (global_verbose) {
                LOG_DEBUG("Primitive draw type set to points");
            }
        } else if (modifier_shift) {
            global_primitive = PRIMITIVE_LINE;
            if (global_verbose) {
                LOG_DEBUG("Primitive draw type set to lines");
            }
        } else {
            global_primitive = PRIMITIVE_TRIANGLE;
            if (global_verbose) {
                LOG_DEBUG("Primitive draw type set to triangles");
            }
        }
    } break;
    #endif /* _DEBUG */

    #ifdef _DEBUG
    case KEY_2: {
        if (modifier_shift) {
            global_shading = SHADING_FLAT;
            if (global_verbose) {
                LOG_DEBUG("Shading set to flat");
            }
        } else {
            global_shading = SHADING_SMOOTH;
            if (global_verbose) {
                LOG_DEBUG("Shading set to smooth");
            }
        }
    } break;
    #endif /* _DEBUG */

    #ifdef _DEBUG
    case KEY_3: {
        if (modifier_control && modifier_shift) {
            global_color = COLOR_SPECULAR;
            if (global_verbose) {
                LOG_DEBUG("Coloring set to specular component only");
            }
        } else if (modifier_shift) {
            global_color = COLOR_DIFFUSE;
            if (global_verbose) {
                LOG_DEBUG("Coloring set to diffuse component only");
            }
        } else {
            global_color = COLOR_AMBIENT;
            if (global_verbose) {
                LOG_DEBUG("Coloring set to ambient component only");
            }
        }
    } break;
    #endif /* _DEBUG */

    #ifdef _DEBUG
    case KEY_4: {
        if (modifier_shift) {
            global_color = COLOR_CAMERA;
            if (global_verbose) {
                LOG_DEBUG("Coloring set to camera reflector illumination");
            }
        } else {
            global_color = COLOR_BLINN_PHONG;
            if (global_verbose) {
                LOG_DEBUG("Coloring set to Blinn-Phong illumination");
            }
        }
    } break;
    #endif /* _DEBUG */

    #ifdef _DEBUG
    case KEY_5: {
        if (modifier_shift) {
            global_color = COLOR_DEPTH;
            if (global_verbose) {
                LOG_DEBUG("Coloring set to depth buffer");
            }
        } else {
            global_color = COLOR_NORMAL;
            if (global_verbose) {
                LOG_DEBUG("Coloring set to normal vectors");
            }
        }
    } break;
    #endif /* _DEBUG */

    case KEY_A: {
        if (modifier_control && modifier_shift) {
            scene_set_camera_longitude(INPUT_CAM_LONGITUDE_DELTA_1);
        } else if (modifier_shift) {
            scene_set_camera_longitude(INPUT_CAM_LONGITUDE_DELTA_2);
        } else {
            scene_set_camera_longitude(INPUT_CAM_LONGITUDE_DELTA_3);
        }
    } break;

    case KEY_B: {
        if (modifier_control && modifier_shift) {
            scene_set_model_rotate_y(SCENE_VALUE_RESET);
        } else if (modifier_shift) {
            scene_set_model_rotate_y(-INPUT_MODEL_ROTATION_DELTA);
        } else {
            scene_set_model_rotate_y(INPUT_MODEL_ROTATION_DELTA);
        }
    } break;

    case KEY_C: {
        if (modifier_control && modifier_shift) {
            scene_set_camera_clip_far(SCENE_VALUE_RESET);
        } else if (modifier_shift) {
            scene_set_camera_clip_far(INPUT_CAM_CLIP_FAR_DELTA);
        } else {
            scene_set_camera_clip_far(-INPUT_CAM_CLIP_FAR_DELTA);
        }
    } break;

    case KEY_D: {
        if (modifier_control && modifier_shift) {
            scene_set_camera_longitude(-INPUT_CAM_LONGITUDE_DELTA_1);
        } else if (modifier_shift) {
            scene_set_camera_longitude(-INPUT_CAM_LONGITUDE_DELTA_2);
        } else {
            scene_set_camera_longitude(-INPUT_CAM_LONGITUDE_DELTA_3);
        }
    } break;

    case KEY_E: {
        if (modifier_control && modifier_shift) {
            scene_set_camera_latitude(INPUT_CAM_LATITUDE_DELTA_1);
        } else if (modifier_shift) {
            scene_set_camera_latitude(INPUT_CAM_LATITUDE_DELTA_2);
        } else {
            scene_set_camera_latitude(INPUT_CAM_LATITUDE_DELTA_3);
        }
    } break;

    case KEY_F: {
        if (modifier_control && modifier_shift) {
            scene_set_model_scale_x(SCENE_VALUE_RESET);
        } else if (modifier_shift) {
            scene_set_model_scale_x(-INPUT_MODEL_SCALE_DELTA);
        } else {
            scene_set_model_scale_x(INPUT_MODEL_SCALE_DELTA);
        }
    } break;

    case KEY_G: {
        if (modifier_control && modifier_shift) {
            scene_set_model_scale_y(SCENE_VALUE_RESET);
        } else if (modifier_shift) {
            scene_set_model_scale_y(-INPUT_MODEL_SCALE_DELTA);
        } else {
            scene_set_model_scale_y(INPUT_MODEL_SCALE_DELTA);
        }
    } break;

    case KEY_H: {
        if (modifier_control && modifier_shift) {
            scene_set_model_scale_z(SCENE_VALUE_RESET);
        } else if (modifier_shift) {
            scene_set_model_scale_z(-INPUT_MODEL_SCALE_DELTA);
        } else {
            scene_set_model_scale_z(INPUT_MODEL_SCALE_DELTA);
        }
    } break;

    case KEY_I: {
        if (modifier_control && modifier_shift) {
            scene_set_light_latitude(SCENE_VALUE_RESET);
        } else if (modifier_shift) {
            scene_set_light_latitude(-INPUT_LIGHT_LATITUDE_DELTA);
        } else {
            scene_set_light_latitude(INPUT_LIGHT_LATITUDE_DELTA);
        }
    } break;

    #ifdef _DEBUG
    case KEY_M: {
        if (modifier_shift) {
            if (global_culling & CULLING_FRUSTUM) {
                global_culling ^= CULLING_FRUSTUM;
                if (global_verbose) {
                    LOG_DEBUG("Disabled view frustum clipping");
                }
            } else {
                global_culling |= CULLING_FRUSTUM;
                if (global_verbose) {
                    LOG_DEBUG("Enabled view frustum clipping");
                }
            }
        } else {
            if (global_culling & CULLING_BACKFACE) {
                global_culling ^= CULLING_BACKFACE;
                if (global_verbose) {
                    LOG_DEBUG("Disabled backface culling");
                }
            } else {
                global_culling |= CULLING_BACKFACE;
                if (global_verbose) {
                    LOG_DEBUG("Enabled backface culling");
                }
            }
        }
    } break;
    #endif /* _DEBUG */

    case KEY_N: {
        if (modifier_control && modifier_shift) {
            scene_set_model_rotate_z(SCENE_VALUE_RESET);
        } else if (modifier_shift) {
            scene_set_model_rotate_z(-INPUT_MODEL_ROTATION_DELTA);
        } else {
            scene_set_model_rotate_z(INPUT_MODEL_ROTATION_DELTA);
        }
    } break;

    case KEY_O: {
        if (modifier_control && modifier_shift) {
            scene_set_light_distance(SCENE_VALUE_RESET);
        } else if (modifier_shift) {
            scene_set_light_distance(-INPUT_LIGHT_DISTANCE_DELTA);
        } else {
            scene_set_light_distance(INPUT_LIGHT_DISTANCE_DELTA);
        }
    } break;

    case KEY_P: {
        if (modifier_shift) {
            scene_set_light_shininess(-INPUT_LIGHT_SHININESS_DELTA);
        } else {
            scene_set_light_shininess(INPUT_LIGHT_SHININESS_DELTA);
        }
    } break;

    case KEY_Q: {
        if (modifier_control && modifier_shift) {
            scene_set_camera_latitude(-INPUT_CAM_LATITUDE_DELTA_1);
        } else if (modifier_shift) {
            scene_set_camera_latitude(-INPUT_CAM_LATITUDE_DELTA_2);
        } else {
            scene_set_camera_latitude(-INPUT_CAM_LATITUDE_DELTA_3);
        }
    } break;

    case KEY_R: {
        if (modifier_control && modifier_shift) {
            scene_set_model_position_x(SCENE_VALUE_RESET);
        } else if (modifier_shift) {
            scene_set_model_position_x(-INPUT_MODEL_POSITION_DELTA);
        } else {
            scene_set_model_position_x(INPUT_MODEL_POSITION_DELTA);
        }
    } break;

    case KEY_S: {
        if (modifier_control && modifier_shift) {
            scene_set_camera_distance(1.0f/INPUT_CAM_DISTANCE_DELTA_1);
        } else if (modifier_shift) {
            scene_set_camera_distance(1.0f/INPUT_CAM_DISTANCE_DELTA_2);
        } else {
            scene_set_camera_distance(1.0f/INPUT_CAM_DISTANCE_DELTA_3);
        }
    } break;

    case KEY_T: {
        if (modifier_control && modifier_shift) {
            scene_set_model_position_y(SCENE_VALUE_RESET);
        } else if (modifier_shift) {
            scene_set_model_position_y(-INPUT_MODEL_POSITION_DELTA);
        } else {
            scene_set_model_position_y(INPUT_MODEL_POSITION_DELTA);
        }
    } break;

    case KEY_U: {
        if (modifier_control && modifier_shift) {
            scene_set_light_longitude(SCENE_VALUE_RESET);
        } else if (modifier_shift) {
            scene_set_light_longitude(-INPUT_LIGHT_LONGITUDE_DELTA);
        } else {
            scene_set_light_longitude(INPUT_LIGHT_LONGITUDE_DELTA);
        }
    } break;

    case KEY_V: {
        if (modifier_control && modifier_shift) {
            scene_set_model_rotate_x(SCENE_VALUE_RESET);
        } else if (modifier_shift) {
            scene_set_model_rotate_x(-INPUT_MODEL_ROTATION_DELTA);
        } else {
            scene_set_model_rotate_x(INPUT_MODEL_ROTATION_DELTA);
        }
    } break;

    case KEY_W: {
        if (modifier_control && modifier_shift) {
            scene_set_camera_distance(INPUT_CAM_DISTANCE_DELTA_1);
        } else if (modifier_shift) {
            scene_set_camera_distance(INPUT_CAM_DISTANCE_DELTA_2);
        } else {
            scene_set_camera_distance(INPUT_CAM_DISTANCE_DELTA_3);
        }
    } break;

    case KEY_X: {
        if (modifier_control && modifier_shift) {
            scene_set_camera_clip_near(SCENE_VALUE_RESET);
        } else if (modifier_shift) {
            scene_set_camera_clip_near(INPUT_CAM_CLIP_NEAR_DELTA);
        } else {
            scene_set_camera_clip_near(1.0f / INPUT_CAM_CLIP_NEAR_DELTA);
        }
    } break;

    case KEY_Y: {
        if (!global_layout) {
            if (modifier_control && modifier_shift) {
                scene_set_model_position_z(SCENE_VALUE_RESET);
            } else if (modifier_shift) {
                scene_set_model_position_z(-INPUT_MODEL_POSITION_DELTA);
            } else {
                scene_set_model_position_z(INPUT_MODEL_POSITION_DELTA);
            }
        } else {
            if (modifier_control && modifier_shift) {
                scene_set_camera_fov(SCENE_VALUE_RESET);
            } else if (modifier_shift) {
                scene_set_camera_fov(INPUT_CAM_FOV_DELTA);
            } else {
                scene_set_camera_fov(-INPUT_CAM_FOV_DELTA);
            }
        }
    } break;

    case KEY_Z: {
        if (!global_layout) {
            if (modifier_control && modifier_shift) {
                scene_set_camera_fov(SCENE_VALUE_RESET);
            } else if (modifier_shift) {
                scene_set_camera_fov(INPUT_CAM_FOV_DELTA);
            } else {
                scene_set_camera_fov(-INPUT_CAM_FOV_DELTA);
            }
        } else {
            if (modifier_control && modifier_shift) {
                scene_set_model_position_z(SCENE_VALUE_RESET);
            } else if (modifier_shift) {
                scene_set_model_position_z(-INPUT_MODEL_POSITION_DELTA);
            } else {
                scene_set_model_position_z(INPUT_MODEL_POSITION_DELTA);
            }
        }
    } break;

    }
} /* input_down */


/* Process key release input */
extern void input_up(
    unsigned long long key // Key
) {
    switch (key) {

    case KEY_SHIFT: {
        modifier_shift = 0;
    } break;

    case KEY_CONTROL: {
        modifier_control = 0;
    } break;

    }
} /* input_up */


/* -------------------------------------------------------------------------- */


/* input.c */