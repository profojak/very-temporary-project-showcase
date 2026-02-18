/********************************************************************
  Frame buffer utilities for MicroZed based MZ_APO board.
********************************************************************/

#define _POSIX_C_SOURCE 200112L
#define BLKW 60
#define BLKH 32

#include "fbuffer.h"

#include <stdio.h>

/* frame buffer init */
fbuffer_t *fb_init(int w, int h)
{
  fbuffer_t *fb = (fbuffer_t *)malloc(sizeof(fbuffer_t));
  fb->w = w;
  fb->h = h;
  fb->r = (char *)malloc(w * h + w * BLKH);
  fb->g = (char *)malloc(w * h + w * BLKH);
  fb->b = (char *)malloc(w * h + w * BLKH);
  memset(fb->r, '0', w * h + w * BLKH);
  memset(fb->g, '0', w * h + w * BLKH);
  memset(fb->b, '0', w * h + w * BLKH);
  return fb;
}

/* move frame buffer 8 pixels up */
void fb_up(fbuffer_t *fb) {
  int i, j;
  for (i = 0; i < fb->h + BLKH - 9; i++) {
    for (j = 0; j < fb->w; j++) {
      fb->r[i * fb->w + j] = fb->r[(i + 8) * fb->w + j];
      fb->g[i * fb->w + j] = fb->g[(i + 8) * fb->w + j];
      fb->b[i * fb->w + j] = fb->b[(i + 8) * fb->w + j];
    }
  }
}

/* fill area of frame buffer */
void fb_fill(int x, int y, int w, int h, char *rgb, fbuffer_t *fb)
{
  int i, j;
  for (i = x; i < x + w; i++) {
    for (j = y; j < y + h; j++) {
      fb->r[i + j * fb->w] = rgb[0];
      fb->g[i + j * fb->w] = rgb[1];
      fb->b[i + j * fb->w] = rgb[2];
    }
  }
}


/* outline area of frame buffer */
void fb_outl(int x, int y, int w, int h, char *rgb, fbuffer_t *fb)
{
  int i, j;
  for (i = x; i < x + w; i++) {
    for (j = y; j < y + h; j++) {
      if ((i < x + 5) || (i > x + w - 5) ||
        (j < y + 5) || (j > y + h - 5)) {
        fb->r[i + j * fb->w] = rgb[0];
        fb->g[i + j * fb->w] = rgb[1];
        fb->b[i + j * fb->w] = rgb[2];
      }
    }
  }
}

/* block to frame buffer */
void fb_block(int x, int y, bool outline, char *rgb, fbuffer_t *fb)
{
  x = x * (BLKW + 20) + 10; // index of music line 0-5
  y = y * BLKH;             // index of block y
  if (outline) {
    fb_outl(x, y, BLKW, BLKH, rgb, fb);
  } else {
    fb_fill(x, y, BLKW, BLKH, rgb, fb);
  }
}

/* text to frame buffer */
void fb_text14x16(int x, int y, char *text, int size, char *rgb,
  fbuffer_t *fb)
{
  int ln, ch, i, j, w, width, n = strlen(text), ptr, offset = 0;
  for (ch = 0; ch < n; ch++) {
    offset += font_winFreeSystem14x16.width[text[ch] - 0x20];
  }
  offset = -(offset * 0.5) * size + x + y * fb->w;
  unsigned short mask = 1 << 15; // mask to get bit
  for (ln = 0; ln < font_winFreeSystem14x16.height; ln++) {
    ptr = ln * fb->w * size;
    for (ch = 0; ch < n; ch++) {
      width = font_winFreeSystem14x16.width[text[ch] - 0x20];
      for (w = 0; w < width; w++) {
        // check for 1 bit with mask
        if (font_winFreeSystem14x16.bits[
          font_winFreeSystem14x16.height * (text[ch] - 0x20) + ln]
          & (short)(mask >> w)) {
          if (offset + ptr + size * fb->w + size < fb->w * fb->h) {
            for (i = 0; i < size; i++) {
              for (j = 0; j < size; j++) {
                fb->r[offset + ptr + i + j * fb->w] = rgb[0];
                fb->g[offset + ptr + i + j * fb->w] = rgb[1];
                fb->b[offset + ptr + i + j * fb->w] = rgb[2];
              }
            }
          }
        }
        ptr += size;
      }
    }
  }
}

/* 2 option menu to frame buffer */
void fb_menu2(menu_t *menu, fbuffer_t *fb)
{
  fb_fill(30, 20, fb->w - 60, fb->h - 40, "000", fb);
  fb_outl(50, 40, fb->w - 100, fb->h - 80, menu->rgb, fb);
  if (menu->active == 0) {
    fb_fill(52, 42, fb->w - 104, (fb->h - 84) / 2, menu->rgb, fb);
    fb_text14x16(fb->w / 2, 60, menu->options[0], 6, "000", fb);
    fb_text14x16(fb->w / 2, 176, menu->options[1], 6, menu->rgb, fb);
  } else {
    fb_fill(52, 42 + (fb->h - 84) / 2, fb->w - 104, (fb->h - 84) / 2,
      menu->rgb, fb);
    fb_text14x16(fb->w / 2, 60, menu->options[0], 6, menu->rgb, fb);
    fb_text14x16(fb->w / 2, 176, menu->options[1], 6, "000", fb);
  }
}

/* 1 option menu to frame buffer */
void fb_menu1(menu_t *menu, fbuffer_t *fb)
{
  fb_fill(30, 20, fb->w - 60, fb->h - 40, "000", fb);
  fb_outl(50, 40, fb->w - 100, fb->h - 80, menu->rgb, fb);
  fb_text14x16(fb->w / 2, 60, menu->options[0], 6, menu->rgb, fb);
  fb_fill(65, 175, 20 + menu->active * 165, 90, menu->rgb, fb);
}
