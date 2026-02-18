#include "raylib.h"
#include "rlgl.h"
#include "raymath.h"

#include <stdlib.h>
#include <string.h>

#include <stdio.h>

#define DARKRED (Color){ 160, 0, 25, 255 }
#define M 120

// Electric
// ----------------------------------------------------------------------------

typedef struct {
    Vector2 pos;
    Rectangle rect;
    int state;
    int goal;
    int frames;
} Light;

typedef struct {
    Vector2 pos;
    Rectangle rect;
    bool on;
} Switch;

typedef struct {
    Vector2 pos;
    Vector2 last_pos;
    int frames;
} Head;

typedef struct OutletI OutletI;

typedef struct {
    Vector2 pos;
    Rectangle rect;
    Head head;
    Color color;
    bool on;
    OutletI* connected;
} OutletO;

typedef struct OutletI {
    Vector2 pos;
    Rectangle rect;
    Color color;
    OutletO* connected;
} OutletI;

OutletO CreateOutletO(Vector2 pos, Color color) {
    OutletO outlet = { 0 };
    outlet.pos = pos;
    outlet.rect = (Rectangle){ pos.x - 50, pos.y - 50, 100, 100 };
    outlet.head.pos = pos;
    outlet.color = color;
    return outlet;
}

OutletI CreateOutletI(Vector2 pos, Color color) {
    OutletI outlet = { 0 };
    outlet.pos = pos;
    outlet.rect = (Rectangle){ pos.x - 50, pos.y - 50, 100, 100 };
    outlet.color = color;
    return outlet;
}

Switch CreateSwitch(Vector2 pos) {
    Switch sw = { 0 };
    sw.pos = pos;
    sw.rect = (Rectangle){ pos.x - 50, pos.y - 50, 100, 100 };
    sw.on = false;
    return sw;
}

Light CreateLight(Vector2 pos, int goal) {
    Light light = { 0 };
    light.pos = pos;
    light.rect = (Rectangle){ pos.x - 50, pos.y - 50, 100, 100 };
    light.goal = goal;
    light.state = 0;
    return light;
}

// Control
// ----------------------------------------------------------------------------

typedef struct {
    OutletO* outlet;
    Vector2 choke_point;
    int frames;
    bool pause;
    bool fail;
} Mouse;

typedef struct {
    Light light;
    Switch* switches;
    OutletO* out_o;
    OutletI* out_i;
    int num_s;
    int num_o;
    int num_i;
} Level;

void AllocateLevel(Level* level, int num_o, int num_i, int num_s) {
    level->switches = (Switch*)malloc(num_s * sizeof(Switch));
    level->out_o = (OutletO*)malloc(num_o * sizeof(OutletO));
    level->out_i = (OutletI*)malloc(num_i * sizeof(OutletI));
    level->num_s = num_s;
    level->num_o = num_o;
    level->num_i = num_i;
    memset(level->switches, 0, num_s * sizeof(Switch));
    memset(level->out_o, 0, num_o * sizeof(OutletO));
    memset(level->out_i, 0, num_i * sizeof(OutletI));
}

void ResetLevel(Level* level, Level* orig) {
    if (level->switches) free(level->switches);
    if (level->out_o) free(level->out_o);
    if (level->out_i) free(level->out_i);
    level->switches = (Switch*)malloc(orig->num_s * sizeof(Switch));
    level->out_o = (OutletO*)malloc(orig->num_o * sizeof(OutletO));
    level->out_i = (OutletI*)malloc(orig->num_i * sizeof(OutletI));
    level->num_s = orig->num_s;
    level->num_o = orig->num_o;
    level->num_i = orig->num_i;
    memcpy(level->switches, orig->switches, orig->num_s * sizeof(Switch));
    memcpy(level->out_o, orig->out_o, orig->num_o * sizeof(OutletO));
    memcpy(level->out_i, orig->out_i, orig->num_i * sizeof(OutletI));
    level->light = orig->light;
}

void ResetCamera(Camera2D* cam, Level* level) {
    int x = 0;
    int y = 0;
    for (int i = 0; i < level->num_i; i++) {
        x += level->out_i[i].pos.x;
        y += level->out_i[i].pos.y;
    }
    for (int i = 0; i < level->num_o; i++) {
        x += level->out_o[i].pos.x;
        y += level->out_o[i].pos.y;
    }
    for (int i = 0; i < level->num_s; i++) {
        x += level->switches[i].pos.x;
        y += level->switches[i].pos.y;
    }
    x += level->light.pos.x;
    y += level->light.pos.y;
    x /= (level->num_i + level->num_o + level->num_s + 1);
    y /= (level->num_i + level->num_o + level->num_s + 1);
    cam->target = (Vector2){ x - GetScreenWidth() / 2, y - GetScreenHeight() / 2 };
}

int CheckCollisionOutletO(Mouse* mouse, Camera2D* cam, Level* level) {
    for (int i = 0; i < level->num_o; i++) {
        if (CheckCollisionPointRec(Vector2Add(GetMousePosition(), cam->target), level->out_o[i].rect)) {
            if (!level->out_o[i].connected) {
                mouse->outlet = &level->out_o[i];
                mouse->outlet->head.frames = 0;
                return 1;
            }
        }
    }
    return 0;
}

int CheckCollisionOutletI(Mouse* mouse, Camera2D* cam, Level* level) {
    for (int i = 0; i < level->num_i; i++) {
        if (CheckCollisionPointRec(Vector2Add(GetMousePosition(), cam->target), level->out_i[i].rect)) {
            return i + 1;
        }
    }
    return 0;
}

int CheckCollisionSwitch(Mouse* mouse, Camera2D* cam, Level* level) {
    for (int i = 0; i < level->num_s; i++) {
        if (CheckCollisionPointRec(Vector2Add(GetMousePosition(), cam->target), level->switches[i].rect)) {
            level->switches[i].on = !level->switches[i].on;
            return i + 1;
        }
    }
    return 0;
}

int CheckVector(Vector2 a, Vector2 b) {
    if (a.x == b.x && a.y == b.y) {
        return 1;
    }
    return 0;
}

int CheckColor(Color a, Color b) {
    if (a.r == b.r && a.g == b.g && a.b == b.b) {
        return 1;
    }
    return 0;
}

int CheckCableCrosses(Mouse* mouse, Level* level) {
    for (int i = 0; i < level->num_o; i++) {
        if (level->out_o[i].connected && level->out_o[i].on) {
            for (int j = 0; j < level->num_o; j++) {
                if (level->out_o[j].connected) {
                    if (i != j) {
                        if (CheckCollisionLines(level->out_o[i].pos, level->out_o[i].head.pos, level->out_o[j].pos, level->out_o[j].head.pos, &mouse->choke_point)) {
                            mouse->pause = true;
                            mouse->fail = true;
                            mouse->frames = 300;
                            return 1;
                        }
                    }
                }
            }
        }
    }
    return 0;
}

void DistributePower(Vector2, bool, Level*, Mouse*);

void DistributePower(Vector2 pos, bool on, Level* level, Mouse* mouse) {
    Vector2 up = Vector2Add(pos, (Vector2){ 0, -M });
    Vector2 down = Vector2Add(pos, (Vector2){ 0, M });
    Vector2 left = Vector2Add(pos, (Vector2){ -M, 0 });
    Vector2 right = Vector2Add(pos, (Vector2){ M, 0 });

    for (int i = 0; i < level->num_o; i++) {
        Vector2 pos = level->out_o[i].pos;
        if (CheckVector(up, pos) || CheckVector(down, pos) ||
            CheckVector(left, pos) || CheckVector(right, pos)) {
            level->out_o[i].on = !level->out_o[i].on;
            if (level->out_o[i].connected) {
                if (level->out_o[i].on) {
                    if (CheckColor(level->out_o[i].connected->color, BLANK) ||
                        CheckColor(level->out_o[i].connected->color, level->out_o[i].color)) {
                        DistributePower(level->out_o[i].connected->pos, true, level, mouse);
                    } else {
                        mouse->pause = true;
                        mouse->fail = true;
                        mouse->choke_point = level->out_o[i].connected->pos;
                        mouse->frames = 300;
                    }
                } else {
                    DistributePower(level->out_o[i].connected->pos, false, level, mouse);
                }
            }
        }
    }

    if (CheckVector(up, level->light.pos) || CheckVector(down, level->light.pos) ||
        CheckVector(left, level->light.pos) || CheckVector(right, level->light.pos)) {
        if (on) {
            level->light.frames = 0;
            level->light.state++;
        } else {
            level->light.frames = 0;
            level->light.state--;
        }
    }
}

float EaseOutCubic(float t) {
    return 1.0 - (1.0 - t) * (1.0 - t) * (1.0 - t);
}

float EaseInCubic(float t) {
    return t * t * t;
}

// Draw
// ----------------------------------------------------------------------------

void DrawOutletO(OutletO* outlet) {
    DrawRectangleRoundedLines(outlet->rect, 0.25, 8, 4, DARKGRAY);
    DrawRectangleRounded(outlet->rect, 0.25, 8, GRAY);
    DrawCircleV(outlet->pos, 20, LIGHTGRAY);
}

void DrawOutletI(OutletI* outlet) {
    DrawRectangleRoundedLines(outlet->rect, 0.25, 8, 4, DARKGRAY);
    if (CheckColor(outlet->color, BLANK)) {
        DrawRectangleRounded(outlet->rect, 0.25, 8, GRAY);
        DrawCircle(outlet->pos.x, outlet->pos.y, 40, DARKGRAY);
    } else if (CheckColor(outlet->color, RED)) {
        DrawRectangleRounded(outlet->rect, 0.25, 8, RED);
        DrawCircle(outlet->pos.x, outlet->pos.y, 40, DARKRED);
    } else if (CheckColor(outlet->color, GREEN)) {
        DrawRectangleRounded(outlet->rect, 0.25, 8, LIME);
        DrawCircle(outlet->pos.x, outlet->pos.y, 40, DARKGREEN);
    } else if (CheckColor(outlet->color, BLUE)) {
        DrawRectangleRounded(outlet->rect, 0.25, 8, BLUE);
        DrawCircle(outlet->pos.x, outlet->pos.y, 40, DARKBLUE);
    }
    DrawCircle(outlet->pos.x - 20, outlet->pos.y, 10, BLACK);
    DrawCircle(outlet->pos.x + 20, outlet->pos.y, 10, BLACK);
}

void DrawSwitch(Switch* sw) {
    DrawRectangleRoundedLines(sw->rect, 0.25, 8, 4, DARKGRAY);
    DrawRectangleRounded(sw->rect, 0.25, 8, GRAY);
    if (sw->on) {
        Rectangle rec1 = { sw->pos.x - 38, sw->pos.y - 15, 76, 30 };
        Rectangle rec2 = { sw->pos.x - 28, sw->pos.y, 56, 58 };
        DrawRectangleRoundedLines(rec1, 0.25, 8, 4, DARKGRAY);
        DrawRectangleRoundedLines(rec2, 0.5, 8, 4, DARKGRAY);
        DrawRectangleRounded(rec1, 0.25, 8, LIGHTGRAY);
        DrawRectangleRounded(rec2, 0.5, 8, WHITE);
    } else {
        Rectangle rec1 = { sw->pos.x - 38, sw->pos.y - 15, 76, 30 };
        Rectangle rec2 = { sw->pos.x - 28, sw->pos.y - 58, 56, 58 };
        DrawRectangleRoundedLines(rec1, 0.25, 8, 4, DARKGRAY);
        DrawRectangleRoundedLines(rec2, 0.5, 8, 4, DARKGRAY);
        DrawRectangleRounded(rec1, 0.25, 8, LIGHTGRAY);
        DrawRectangleRounded(rec2, 0.5, 8, WHITE);
    }
}

void DrawLight(Light* light) {
    DrawRectangleRoundedLines(light->rect, 0.25, 8, 4, DARKGRAY);
    DrawRectangleRounded(light->rect, 0.25, 8, GRAY);
    DrawCircleV(light->pos, 42, BLACK);

    if (light->frames == 0) {
        DrawCircleV(light->pos, 38, DARKGRAY);
        DrawCircleV(light->pos, 20, BLACK);
    } else if (light->frames < 3) {
        DrawCircleV(light->pos, 38, GOLD);
        DrawCircleV(light->pos, 32, YELLOW);
        DrawCircleV(light->pos, 20, WHITE);
    } else if (light->frames < 25) {
        DrawCircleV(light->pos, 38, DARKGRAY);
        DrawCircleV(light->pos, 20, BLACK);
    } else if (light->frames < 28) {
        DrawCircleV(light->pos, 38, GOLD);
        DrawCircleV(light->pos, 32, YELLOW);
        DrawCircleV(light->pos, 20, WHITE);
    } else if (light->frames < 35) {
        DrawCircleV(light->pos, 38, DARKGRAY);
        DrawCircleV(light->pos, 20, BLACK);
    } else if (light->frames < 40) {
        DrawCircleV(light->pos, 38, GOLD);
        DrawCircleV(light->pos, 32, YELLOW);
        DrawCircleV(light->pos, 20, WHITE);
    } else if (light->frames < 43) {
        DrawCircleV(light->pos, 38, DARKGRAY);
        DrawCircleV(light->pos, 20, BLACK);
    } else if (light->frames < 55) {
        DrawCircleV(light->pos, 38, GOLD);
        DrawCircleV(light->pos, 32, YELLOW);
        DrawCircleV(light->pos, 20, WHITE);
    } else if (light->frames < 58) {
        DrawCircleV(light->pos, 38, DARKGRAY);
        DrawCircleV(light->pos, 20, BLACK);
    } else if (light->frames < 64) {
        DrawCircleV(light->pos, 38, GOLD);
        DrawCircleV(light->pos, 32, YELLOW);
        DrawCircleV(light->pos, 20, WHITE);
    } else if (light->frames < 67) {
        DrawCircleV(light->pos, 38, DARKGRAY);
        DrawCircleV(light->pos, 20, BLACK);
    } else if (light->frames < 86) {
        DrawCircleV(light->pos, 38, GOLD);
        DrawCircleV(light->pos, 32, YELLOW);
        DrawCircleV(light->pos, 20, WHITE);
    } else if (light->frames < 89) {
        DrawCircleV(light->pos, 38, DARKGRAY);
        DrawCircleV(light->pos, 20, BLACK);
    } else if (light->state == light->goal) {
        DrawCircleV(light->pos, 38, GOLD);
        DrawCircleV(light->pos, 32, YELLOW);
        DrawCircleV(light->pos, 20, WHITE);
    } else {
        DrawCircleV(light->pos, 38, DARKGRAY);
        DrawCircleV(light->pos, 20, BLACK);
    }
}

void DrawHead(OutletO* outlet) {
    Vector2 pos = outlet->head.pos;
    if (outlet->head.frames) {
        pos = Vector2Lerp(outlet->head.last_pos, outlet->head.pos, EaseOutCubic((float)(30 - outlet->head.frames) / 30));
    }

    if (CheckColor(outlet->color, RED)) {
        DrawCircleV(pos, 42, DARKRED);
        DrawCircleV(pos, 38, RED);
        DrawCircleV(pos, 20, ORANGE);
    } else if (CheckColor(outlet->color, GREEN)) {
        DrawCircleV(pos, 42, DARKGREEN);
        DrawCircleV(pos, 38, LIME);
        DrawCircleV(pos, 20, GREEN);
    } else if (CheckColor(outlet->color, BLUE)) {
        DrawCircleV(pos, 42, DARKBLUE);
        DrawCircleV(pos, 38, BLUE);
        DrawCircleV(pos, 20, SKYBLUE);
    }
}

void DrawCable(OutletO* outlet) {
    Vector2 pos = outlet->head.pos;
    if (outlet->head.frames) {
        pos = Vector2Lerp(outlet->head.last_pos, outlet->head.pos, EaseOutCubic((float)(30 - outlet->head.frames) / 30));
    }

    DrawCircleV(outlet->pos, 6, BLACK);
    DrawCircleV(pos, 6, BLACK);
    DrawLineEx(outlet->pos, pos, 12, BLACK);
}

void DrawChokePoint(Mouse* mouse) {
    if (mouse->frames > 296) {
        DrawLineEx(mouse->choke_point, Vector2Add(mouse->choke_point, (Vector2){ 20, 80 }), 12, GOLD);
        DrawLineEx(Vector2Add(mouse->choke_point, (Vector2){ 20, 80 }), Vector2Add(mouse->choke_point, (Vector2){ 50, 70 }), 12, GOLD);
        DrawLineEx(Vector2Add(mouse->choke_point, (Vector2){ 50, 70 }), Vector2Add(mouse->choke_point, (Vector2){ 70, 140 }), 12, GOLD);
        DrawLineEx(mouse->choke_point, Vector2Add(mouse->choke_point, (Vector2){ 20, 80 }), 6, YELLOW);
        DrawLineEx(Vector2Add(mouse->choke_point, (Vector2){ 20, 80 }), Vector2Add(mouse->choke_point, (Vector2){ 50, 70 }), 6, YELLOW);
        DrawLineEx(Vector2Add(mouse->choke_point, (Vector2){ 50, 70 }), Vector2Add(mouse->choke_point, (Vector2){ 70, 140 }), 6, YELLOW);
    } else if (mouse->frames > 292) {
        DrawLineEx(mouse->choke_point, Vector2Add(mouse->choke_point, (Vector2){ -90, 0 }), 12, GOLD);
        DrawLineEx(Vector2Add(mouse->choke_point, (Vector2){ -90, 0 }), Vector2Add(mouse->choke_point, (Vector2){ -80, -25 }), 12, GOLD);
        DrawLineEx(Vector2Add(mouse->choke_point, (Vector2){ -80, -25 }), Vector2Add(mouse->choke_point, (Vector2){ -140, -20 }), 12, GOLD);
        DrawLineEx(mouse->choke_point, Vector2Add(mouse->choke_point, (Vector2){ -90, 0 }), 6, YELLOW);
        DrawLineEx(Vector2Add(mouse->choke_point, (Vector2){ -90, 0 }), Vector2Add(mouse->choke_point, (Vector2){ -80, -25 }), 6, YELLOW);
        DrawLineEx(Vector2Add(mouse->choke_point, (Vector2){ -80, -25 }), Vector2Add(mouse->choke_point, (Vector2){ -140, -20 }), 6, YELLOW);
    } else if (mouse->frames > 288) {
        DrawLineEx(mouse->choke_point, Vector2Add(mouse->choke_point, (Vector2){ 50, -50 }), 12, GOLD);
        DrawLineEx(Vector2Add(mouse->choke_point, (Vector2){ 50, -50 }), Vector2Add(mouse->choke_point, (Vector2){ 40, -65 }), 12, GOLD);
        DrawLineEx(Vector2Add(mouse->choke_point, (Vector2){ 40, -65 }), Vector2Add(mouse->choke_point, (Vector2){ 110, -105 }), 12, GOLD);
        DrawLineEx(mouse->choke_point, Vector2Add(mouse->choke_point, (Vector2){ 50, -50 }), 6, YELLOW);
        DrawLineEx(Vector2Add(mouse->choke_point, (Vector2){ 50, -50 }), Vector2Add(mouse->choke_point, (Vector2){ 40, -65 }), 6, YELLOW);
        DrawLineEx(Vector2Add(mouse->choke_point, (Vector2){ 40, -65 }), Vector2Add(mouse->choke_point, (Vector2){ 110, -105 }), 6, YELLOW);
    } else if (mouse->frames > 284) {
        DrawLineEx(mouse->choke_point, Vector2Add(mouse->choke_point, (Vector2){ 20, 80 }), 12, GOLD);
        DrawLineEx(Vector2Add(mouse->choke_point, (Vector2){ 20, 80 }), Vector2Add(mouse->choke_point, (Vector2){ 50, 70 }), 12, GOLD);
        DrawLineEx(Vector2Add(mouse->choke_point, (Vector2){ 50, 70 }), Vector2Add(mouse->choke_point, (Vector2){ 70, 140 }), 12, GOLD);
        DrawLineEx(mouse->choke_point, Vector2Add(mouse->choke_point, (Vector2){ 20, 80 }), 6, YELLOW);
        DrawLineEx(Vector2Add(mouse->choke_point, (Vector2){ 20, 80 }), Vector2Add(mouse->choke_point, (Vector2){ 50, 70 }), 6, YELLOW);
        DrawLineEx(Vector2Add(mouse->choke_point, (Vector2){ 50, 70 }), Vector2Add(mouse->choke_point, (Vector2){ 70, 140 }), 6, YELLOW);
    } else if (mouse->frames > 280) {
        DrawLineEx(mouse->choke_point, Vector2Add(mouse->choke_point, (Vector2){ -50, -50 }), 12, GOLD);
        DrawLineEx(Vector2Add(mouse->choke_point, (Vector2){ -50, -50 }), Vector2Add(mouse->choke_point, (Vector2){ -40, -65 }), 12, GOLD);
        DrawLineEx(Vector2Add(mouse->choke_point, (Vector2){ -40, -65 }), Vector2Add(mouse->choke_point, (Vector2){ -110, -105 }), 12, GOLD);
        DrawLineEx(mouse->choke_point, Vector2Add(mouse->choke_point, (Vector2){ 50, -50 }), 6, YELLOW);
        DrawLineEx(Vector2Add(mouse->choke_point, (Vector2){ -50, -50 }), Vector2Add(mouse->choke_point, (Vector2){ -40, -65 }), 6, YELLOW);
        DrawLineEx(Vector2Add(mouse->choke_point, (Vector2){ -40, -65 }), Vector2Add(mouse->choke_point, (Vector2){ -110, -105 }), 6, YELLOW);
    } else if (mouse->frames > 240) {
    } else if (mouse->frames > 236) {
        DrawLineEx(mouse->choke_point, Vector2Add(mouse->choke_point, (Vector2){ 50, -50 }), 12, GOLD);
        DrawLineEx(Vector2Add(mouse->choke_point, (Vector2){ 50, -50 }), Vector2Add(mouse->choke_point, (Vector2){ 40, -65 }), 12, GOLD);
        DrawLineEx(Vector2Add(mouse->choke_point, (Vector2){ 40, -65 }), Vector2Add(mouse->choke_point, (Vector2){ 110, -105 }), 12, GOLD);
        DrawLineEx(mouse->choke_point, Vector2Add(mouse->choke_point, (Vector2){ 50, -50 }), 6, YELLOW);
        DrawLineEx(Vector2Add(mouse->choke_point, (Vector2){ 50, -50 }), Vector2Add(mouse->choke_point, (Vector2){ 40, -65 }), 6, YELLOW);
        DrawLineEx(Vector2Add(mouse->choke_point, (Vector2){ 40, -65 }), Vector2Add(mouse->choke_point, (Vector2){ 110, -105 }), 6, YELLOW);
    } else if (mouse->frames > 232) {
        DrawLineEx(mouse->choke_point, Vector2Add(mouse->choke_point, (Vector2){ 20, 80 }), 12, GOLD);
        DrawLineEx(Vector2Add(mouse->choke_point, (Vector2){ 20, 80 }), Vector2Add(mouse->choke_point, (Vector2){ 50, 70 }), 12, GOLD);
        DrawLineEx(Vector2Add(mouse->choke_point, (Vector2){ 50, 70 }), Vector2Add(mouse->choke_point, (Vector2){ 70, 140 }), 12, GOLD);
        DrawLineEx(mouse->choke_point, Vector2Add(mouse->choke_point, (Vector2){ 20, 80 }), 6, YELLOW);
        DrawLineEx(Vector2Add(mouse->choke_point, (Vector2){ 20, 80 }), Vector2Add(mouse->choke_point, (Vector2){ 50, 70 }), 6, YELLOW);
        DrawLineEx(Vector2Add(mouse->choke_point, (Vector2){ 50, 70 }), Vector2Add(mouse->choke_point, (Vector2){ 70, 140 }), 6, YELLOW);
    } else if (mouse->frames > 228) {
        DrawLineEx(mouse->choke_point, Vector2Add(mouse->choke_point, (Vector2){ 90, 0 }), 12, GOLD);
        DrawLineEx(Vector2Add(mouse->choke_point, (Vector2){ 90, 0 }), Vector2Add(mouse->choke_point, (Vector2){ 80, -25 }), 12, GOLD);
        DrawLineEx(Vector2Add(mouse->choke_point, (Vector2){ 80, -25 }), Vector2Add(mouse->choke_point, (Vector2){ 140, -20 }), 12, GOLD);
        DrawLineEx(mouse->choke_point, Vector2Add(mouse->choke_point, (Vector2){ 90, 0 }), 6, YELLOW);
        DrawLineEx(Vector2Add(mouse->choke_point, (Vector2){ 90, 0 }), Vector2Add(mouse->choke_point, (Vector2){ 80, -25 }), 6, YELLOW);
        DrawLineEx(Vector2Add(mouse->choke_point, (Vector2){ 80, -25 }), Vector2Add(mouse->choke_point, (Vector2){ 140, -20 }), 6, YELLOW);
    } else if (mouse->frames > 224) {
        DrawLineEx(mouse->choke_point, Vector2Add(mouse->choke_point, (Vector2){ 50, -50 }), 12, GOLD);
        DrawLineEx(Vector2Add(mouse->choke_point, (Vector2){ 50, -50 }), Vector2Add(mouse->choke_point, (Vector2){ 40, -65 }), 12, GOLD);
        DrawLineEx(Vector2Add(mouse->choke_point, (Vector2){ 40, -65 }), Vector2Add(mouse->choke_point, (Vector2){ 110, -105 }), 12, GOLD);
        DrawLineEx(mouse->choke_point, Vector2Add(mouse->choke_point, (Vector2){ 50, -50 }), 6, YELLOW);
        DrawLineEx(Vector2Add(mouse->choke_point, (Vector2){ 50, -50 }), Vector2Add(mouse->choke_point, (Vector2){ 40, -65 }), 6, YELLOW);
        DrawLineEx(Vector2Add(mouse->choke_point, (Vector2){ 40, -65 }), Vector2Add(mouse->choke_point, (Vector2){ 110, -105 }), 6, YELLOW);
    } else if (mouse->frames > 220) {
        DrawLineEx(mouse->choke_point, Vector2Add(mouse->choke_point, (Vector2){ 20, 80 }), 12, GOLD);
        DrawLineEx(Vector2Add(mouse->choke_point, (Vector2){ 20, 80 }), Vector2Add(mouse->choke_point, (Vector2){ 50, 70 }), 12, GOLD);
        DrawLineEx(Vector2Add(mouse->choke_point, (Vector2){ 50, 70 }), Vector2Add(mouse->choke_point, (Vector2){ 70, 140 }), 12, GOLD);
        DrawLineEx(mouse->choke_point, Vector2Add(mouse->choke_point, (Vector2){ 20, 80 }), 6, YELLOW);
        DrawLineEx(Vector2Add(mouse->choke_point, (Vector2){ 20, 80 }), Vector2Add(mouse->choke_point, (Vector2){ 50, 70 }), 6, YELLOW);
        DrawLineEx(Vector2Add(mouse->choke_point, (Vector2){ 50, 70 }), Vector2Add(mouse->choke_point, (Vector2){ 70, 140 }), 6, YELLOW);
    } else if (mouse->frames > 174) {
    } else if (mouse->frames > 170) {
        DrawLineEx(mouse->choke_point, Vector2Add(mouse->choke_point, (Vector2){ -90, 0 }), 12, GOLD);
        DrawLineEx(Vector2Add(mouse->choke_point, (Vector2){ -90, 0 }), Vector2Add(mouse->choke_point, (Vector2){ -80, -25 }), 12, GOLD);
        DrawLineEx(Vector2Add(mouse->choke_point, (Vector2){ -80, -25 }), Vector2Add(mouse->choke_point, (Vector2){ -140, -20 }), 12, GOLD);
        DrawLineEx(mouse->choke_point, Vector2Add(mouse->choke_point, (Vector2){ -90, 0 }), 6, YELLOW);
        DrawLineEx(Vector2Add(mouse->choke_point, (Vector2){ -90, 0 }), Vector2Add(mouse->choke_point, (Vector2){ -80, -25 }), 6, YELLOW);
        DrawLineEx(Vector2Add(mouse->choke_point, (Vector2){ -80, -25 }), Vector2Add(mouse->choke_point, (Vector2){ -140, -20 }), 6, YELLOW);
    } else if (mouse->frames > 166) {
        DrawLineEx(mouse->choke_point, Vector2Add(mouse->choke_point, (Vector2){ -50, -50 }), 12, GOLD);
        DrawLineEx(Vector2Add(mouse->choke_point, (Vector2){ -50, -50 }), Vector2Add(mouse->choke_point, (Vector2){ -40, -65 }), 12, GOLD);
        DrawLineEx(Vector2Add(mouse->choke_point, (Vector2){ -40, -65 }), Vector2Add(mouse->choke_point, (Vector2){ -110, -105 }), 12, GOLD);
        DrawLineEx(mouse->choke_point, Vector2Add(mouse->choke_point, (Vector2){ 50, -50 }), 6, YELLOW);
        DrawLineEx(Vector2Add(mouse->choke_point, (Vector2){ -50, -50 }), Vector2Add(mouse->choke_point, (Vector2){ -40, -65 }), 6, YELLOW);
        DrawLineEx(Vector2Add(mouse->choke_point, (Vector2){ -40, -65 }), Vector2Add(mouse->choke_point, (Vector2){ -110, -105 }), 6, YELLOW);
    } else if (mouse->frames > 162) {
        DrawLineEx(mouse->choke_point, Vector2Add(mouse->choke_point, (Vector2){ 20, 80 }), 12, GOLD);
        DrawLineEx(Vector2Add(mouse->choke_point, (Vector2){ 20, 80 }), Vector2Add(mouse->choke_point, (Vector2){ 50, 70 }), 12, GOLD);
        DrawLineEx(Vector2Add(mouse->choke_point, (Vector2){ 50, 70 }), Vector2Add(mouse->choke_point, (Vector2){ 70, 140 }), 12, GOLD);
        DrawLineEx(mouse->choke_point, Vector2Add(mouse->choke_point, (Vector2){ 20, 80 }), 6, YELLOW);
        DrawLineEx(Vector2Add(mouse->choke_point, (Vector2){ 20, 80 }), Vector2Add(mouse->choke_point, (Vector2){ 50, 70 }), 6, YELLOW);
        DrawLineEx(Vector2Add(mouse->choke_point, (Vector2){ 50, 70 }), Vector2Add(mouse->choke_point, (Vector2){ 70, 140 }), 6, YELLOW);
    } else if (mouse->frames > 158) {
        DrawLineEx(mouse->choke_point, Vector2Add(mouse->choke_point, (Vector2){ 20, 80 }), 12, GOLD);
        DrawLineEx(Vector2Add(mouse->choke_point, (Vector2){ 20, 80 }), Vector2Add(mouse->choke_point, (Vector2){ 50, 70 }), 12, GOLD);
        DrawLineEx(Vector2Add(mouse->choke_point, (Vector2){ 50, 70 }), Vector2Add(mouse->choke_point, (Vector2){ 70, 140 }), 12, GOLD);
        DrawLineEx(mouse->choke_point, Vector2Add(mouse->choke_point, (Vector2){ 20, 80 }), 6, YELLOW);
        DrawLineEx(Vector2Add(mouse->choke_point, (Vector2){ 20, 80 }), Vector2Add(mouse->choke_point, (Vector2){ 50, 70 }), 6, YELLOW);
        DrawLineEx(Vector2Add(mouse->choke_point, (Vector2){ 50, 70 }), Vector2Add(mouse->choke_point, (Vector2){ 70, 140 }), 6, YELLOW);
    } else if (mouse->frames > 154) {
        DrawLineEx(mouse->choke_point, Vector2Add(mouse->choke_point, (Vector2){ 50, -50 }), 12, GOLD);
        DrawLineEx(Vector2Add(mouse->choke_point, (Vector2){ 50, -50 }), Vector2Add(mouse->choke_point, (Vector2){ 40, -65 }), 12, GOLD);
        DrawLineEx(Vector2Add(mouse->choke_point, (Vector2){ 40, -65 }), Vector2Add(mouse->choke_point, (Vector2){ 110, -105 }), 12, GOLD);
        DrawLineEx(mouse->choke_point, Vector2Add(mouse->choke_point, (Vector2){ 50, -50 }), 6, YELLOW);
        DrawLineEx(Vector2Add(mouse->choke_point, (Vector2){ 50, -50 }), Vector2Add(mouse->choke_point, (Vector2){ 40, -65 }), 6, YELLOW);
        DrawLineEx(Vector2Add(mouse->choke_point, (Vector2){ 40, -65 }), Vector2Add(mouse->choke_point, (Vector2){ 110, -105 }), 6, YELLOW);
    }
}

// Main
// ----------------------------------------------------------------------------

int main(void)
{
    InitWindow(GetScreenWidth(), GetScreenHeight(), "Elektrika");
    ToggleFullscreen();
    SetTargetFPS(60);

    Camera2D camera = { 0 };
    camera.zoom = 1.0f;
    Mouse mouse = { 0 };
    mouse.pause = true;
    mouse.fail = false;
    mouse.frames = 0;

    int current_level = 0;
    int frames = 120;
    Vector2 cam_start;
    Vector2 cam_end;

    Level levels[8] = { 0 };
    AllocateLevel(&levels[1], 0, 0, 1);
    levels[1].switches[0] = CreateSwitch((Vector2){ 0, 0 });
    levels[1].light = CreateLight((Vector2){ M, 0 }, 1);
    AllocateLevel(&levels[2], 1, 1, 1);
    levels[2].out_o[0] = CreateOutletO((Vector2){ 0, 0 }, BLUE);
    levels[2].out_i[0] = CreateOutletI((Vector2){ 3*M, M }, BLANK);
    levels[2].switches[0] = CreateSwitch((Vector2){ 0, M });
    levels[2].light = CreateLight((Vector2){ 3*M, 0 }, 1);
    AllocateLevel(&levels[3], 2, 1, 1);
    levels[3].out_o[0] = CreateOutletO((Vector2){ -M, 0 }, BLUE);
    levels[3].out_o[1] = CreateOutletO((Vector2){ M, 0 }, GREEN);
    levels[3].out_i[0] = CreateOutletI((Vector2){ -M, 3*M }, GREEN);
    levels[3].switches[0] = CreateSwitch((Vector2){ 0, 0 });
    levels[3].light = CreateLight((Vector2){ 0, 3*M }, 1);
    AllocateLevel(&levels[4], 2, 2, 1);
    levels[4].out_o[0] = CreateOutletO((Vector2){ 0, -M }, RED);
    levels[4].out_o[1] = CreateOutletO((Vector2){ -3*M, 0 }, GREEN);
    levels[4].out_i[0] = CreateOutletI((Vector2){ -3*M, M }, BLANK);
    levels[4].out_i[1] = CreateOutletI((Vector2){ -6*M, -M }, GREEN);
    levels[4].switches[0] = CreateSwitch((Vector2){ 0, 0 });
    levels[4].light = CreateLight((Vector2){ -6*M, 0 }, 1);
    AllocateLevel(&levels[5], 2, 2, 2);
    levels[5].light = CreateLight((Vector2){ 0, 0 }, 2);
    levels[5].out_i[0] = CreateOutletI((Vector2){ -M, 0 }, BLANK);
    levels[5].out_i[1] = CreateOutletI((Vector2){ M, 0 }, BLANK);
    levels[5].out_o[0] = CreateOutletO((Vector2){ -2*M, -3*M }, BLUE);
    levels[5].out_o[1] = CreateOutletO((Vector2){ 2*M, -3*M }, BLUE);
    levels[5].switches[0] = CreateSwitch((Vector2){ -M, -3*M });
    levels[5].switches[1] = CreateSwitch((Vector2){ M, -3*M });
    AllocateLevel(&levels[6], 3, 3, 1);
    levels[6].light = CreateLight((Vector2){ 0, 0 }, 2);
    levels[6].out_i[0] = CreateOutletI((Vector2){ 0, M }, BLANK);
    levels[6].out_i[1] = CreateOutletI((Vector2){ M, 0 }, BLANK);
    levels[6].out_i[2] = CreateOutletI((Vector2){ 3*M, 0 }, BLANK);
    levels[6].out_o[0] = CreateOutletO((Vector2){ 3*M, M }, RED);
    levels[6].out_o[1] = CreateOutletO((Vector2){ 3*M, -M }, GREEN);
    levels[6].out_o[2] = CreateOutletO((Vector2){ 5*M, -M }, BLUE);
    levels[6].switches[0] = CreateSwitch((Vector2){ 5*M, 0 });
    AllocateLevel(&levels[7], 3, 3, 1);
    levels[7].light = CreateLight((Vector2){ M, -M }, 1);
    levels[7].switches[0] = CreateSwitch((Vector2){ 0, 0 });
    levels[7].out_i[0] = CreateOutletI((Vector2){ 0, -M }, BLUE);
    levels[7].out_i[1] = CreateOutletI((Vector2){ -2*M, M }, BLUE);
    levels[7].out_i[2] = CreateOutletI((Vector2){ 2*M, 0 }, BLANK);
    levels[7].out_o[0] = CreateOutletO((Vector2){ -M, 0 }, RED);
    levels[7].out_o[1] = CreateOutletO((Vector2){ 2*M, M }, BLUE);
    levels[7].out_o[2] = CreateOutletO((Vector2){ -2*M, 0 }, BLUE);

    Level* level = (Level*)malloc(sizeof(Level));
    memset(level, 0, sizeof(Level));
    ResetLevel(level, &levels[current_level]);

    while (!WindowShouldClose()) {

        // Cinematic
        // --------------------------------------------------------------------

        if (mouse.pause && current_level < 8) {
            if (!mouse.fail) {
                if (frames > 120) {
                    cam_start = camera.target;
                    cam_end = Vector2Add(cam_start, (Vector2){ 0, 1.5 * GetScreenHeight() });
                    camera.target = Vector2Lerp(cam_start, cam_end, EaseInCubic(1.0 - (float)(frames - 120) / 120.0));
                } else if (frames == 120) {
                    current_level++;
                    if (current_level == 8) {

                    } else {
                        ResetLevel(level, &levels[current_level]);
                        ResetCamera(&camera, level);
                        cam_end = camera.target;
                        cam_start = Vector2Add(cam_end, (Vector2){ 0, -1.5 * GetScreenHeight() });
                        camera.target = cam_start;
                    }
                } else if (frames > 0) {
                    camera.target = Vector2Lerp(cam_start, cam_end, EaseOutCubic((float)(120 - frames) / 120));
                } else if (frames == 0) {
                    camera.target = cam_end;
                    mouse.pause = false;
                }
                frames--;
            } else {
                if (mouse.frames > 120) {
                    mouse.frames--;
                } else if (mouse.frames > 0) {
                    cam_start = camera.target;
                    cam_end = Vector2Add(cam_start, (Vector2){ 0, 1.5 * GetScreenHeight() });
                    camera.target = Vector2Lerp(cam_start, cam_end, EaseInCubic(1.0 - (float)(mouse.frames) / 120.0));
                    mouse.frames--;
                } else if (mouse.frames == 0) {
                    current_level--;
                    mouse.fail = false;
                    frames = 120;
                    mouse.frames = 0;
                }
            }
        }

        // Update
        // --------------------------------------------------------------------
        
        if (current_level < 8) {
        if (IsMouseButtonDown(MOUSE_BUTTON_RIGHT) && !mouse.pause) {
            Vector2 delta = GetMouseDelta();
            camera.target = Vector2Add(camera.target, Vector2Scale(delta, -1));
        } else if (IsMouseButtonDown(MOUSE_BUTTON_LEFT) && !mouse.pause) {
            Vector2 pos = Vector2Add(GetMousePosition(), camera.target);
            if (mouse.outlet) {
                mouse.outlet->head.pos = pos;
            } else {
                if (!CheckCollisionOutletO(&mouse, &camera, level)) {
                    int i = CheckCollisionOutletI(&mouse, &camera, level);
                    if (i && level->out_i[i - 1].connected) {
                        mouse.outlet = level->out_i[i - 1].connected;
                        if (level->out_i[i - 1].connected->on) {
                            DistributePower(level->out_i[i - 1].pos, false, level, &mouse);
                        }
                        level->out_i[i - 1].connected = 0;
                    }
                }
                if (CheckCableCrosses(&mouse, level)) {
                    level->light.frames = 0;
                    level->light.state = 0;
                }
            }
        } else if (IsMouseButtonReleased(MOUSE_BUTTON_LEFT) && !mouse.pause) {
            if(mouse.outlet) {
                int i = CheckCollisionOutletI(&mouse, &camera, level);
                if (i && !level->out_i[i - 1].connected) {
                    mouse.outlet->connected = &level->out_i[i - 1];
                    mouse.outlet->head.pos = level->out_i[i - 1].pos;
                    level->out_i[i - 1].connected = mouse.outlet;
                    if (mouse.outlet->on) {
                        if (CheckColor(mouse.outlet->connected->color, BLANK) ||
                            CheckColor(mouse.outlet->connected->color, mouse.outlet->color)) {
                            DistributePower(mouse.outlet->connected->pos, true, level, &mouse);
                        } else {
                            mouse.pause = true;
                            mouse.fail = true;
                            mouse.choke_point = mouse.outlet->connected->pos;
                            mouse.frames = 300;
                        }
                    }
                } else {
                    mouse.outlet->connected = false;
                    mouse.outlet->head.last_pos = mouse.outlet->head.pos;
                    mouse.outlet->head.pos = mouse.outlet->pos;
                    mouse.outlet->head.frames = 30;
                }
                mouse.outlet = 0;

                if (CheckCableCrosses(&mouse, level)) {
                    level->light.frames = 0;
                    level->light.state = 0;
                }
            } else {
                int s = CheckCollisionSwitch(&mouse, &camera, level);
                if (s) {
                    DistributePower(level->switches[s - 1].pos, level->switches[s - 1].on, level, &mouse);
                }

                if (CheckCableCrosses(&mouse, level)) {
                    level->light.frames = 0;
                    level->light.state = 0;
                }
            }
        }

        for (int i = 0; i < level->num_o; i++) {
            if (level->out_o[i].head.frames) {
                level->out_o[i].head.frames--;
            }
        }

        if (level->light.state) {
            if (level->light.state == level->light.goal) {
                if (level->light.frames == 150) {
                    mouse.pause = true;
                    frames = 240;
                }
            } else if (level->light.frames == 150) {
                level->light.frames = 0;
            }
            level->light.frames++;
        }
        }
        
        // Draw
        // --------------------------------------------------------------------
        
        BeginDrawing();
            ClearBackground(RAYWHITE);

            if (current_level == 8) {

                DrawText("Congratulations!", GetScreenWidth() / 2 - MeasureText("Congratulations!", 40) / 2, GetScreenHeight() / 2 - 40, 40, BLACK);
            
            } else {
            BeginMode2D(camera);
                for (int i = 0; i < level->num_i; i++) {
                    DrawOutletI(&level->out_i[i]);
                }
                for (int i = 0; i < level->num_o; i++) {
                    DrawOutletO(&level->out_o[i]);
                }
                for (int i = 0; i < level->num_s; i++) {
                    DrawSwitch(&level->switches[i]);
                }
                DrawLight(&level->light);
                for (int i = 0; i < level->num_o; i++) {
                    DrawHead(&level->out_o[i]);
                }
                for (int i = 0; i < level->num_o; i++) {
                    DrawCable(&level->out_o[i]);
                }
                DrawChokePoint(&mouse);
            EndMode2D();
            }
            
        EndDrawing();
    }

    CloseWindow();
    return 0;
}