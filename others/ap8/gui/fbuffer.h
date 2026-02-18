/********************************************************************
  Frame buffer utilities for MicroZed based MZ_APO board.
********************************************************************/

#ifndef FBUFFER_H
#define FBUFFER_H

#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdbool.h>

#include "../fonts/font_types.h"
#include "menu.h"

/* frame buffer structure for better color manipulation */
typedef struct {
  unsigned int w, h; /* width and height */
  char *r, *g, *b;   /* arrays for red, green and blue */
} fbuffer_t;

fbuffer_t *fb_init(int w, int h);

void fb_up(fbuffer_t *fb);

void fb_fill(int x, int y, int w, int h, char *rgb, fbuffer_t *fb);

void fb_outl(int x, int y, int w, int h, char *rgb, fbuffer_t *fb);

void fb_block(int x, int y, bool outline, char *rgb, fbuffer_t *fb);

void fb_text14x16(int x, int y, char *text, int size, char *rgb,
  fbuffer_t *fb);

void fb_menu2(menu_t *menu, fbuffer_t *fb);

void fb_menu1(menu_t *menu, fbuffer_t *fb);

#endif /*FBUFFER_H*/
