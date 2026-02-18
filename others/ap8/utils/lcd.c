/********************************************************************
  LCD utilities for MicroZed based MZ_APO board.
********************************************************************/

#define _POSIX_C_SOURCE 200112L

#include "lcd.h"

/* LCD init */
void lcd_init(unsigned char *lcd_mem_base, unsigned int size)
{
  parlcd_hx8357_init(lcd_mem_base);
  parlcd_write_cmd(lcd_mem_base, 0x2c);
  int i;
  uint16_t c;
  for (i = 0; i < size; i++) {
    c = 0;
    parlcd_write_data(lcd_mem_base, c);
  }
}

/* draw frame buffer to LCD */
void lcd_draw(unsigned char *lcd_mem_base, fbuffer_t *fb)
{
  int i, j, ptr = 0;
  char r, g, b;
  uint16_t c;
  parlcd_write_cmd(lcd_mem_base, 0x2c);
  for (i = 0; i < fb->w; i++) {
    for (j = 0; j < fb->h; j++) {
      r = fb->r[ptr] > 57 ? fb->r[ptr] - 88 : fb->r[ptr] - 48;
      g = fb->g[ptr] > 57 ? fb->g[ptr] - 88 : fb->g[ptr] - 48;
      b = fb->b[ptr] > 57 ? fb->b[ptr] - 88 : fb->b[ptr] - 48;
      c = 0x0 + ((r * 2) << 11) + ((g * 2) << 5) + b * 2;
      ptr++;
      parlcd_write_data(lcd_mem_base, c);
    }
  }
}

/* splash screen */
void lcd_splash(unsigned char *lcd_mem_base, fbuffer_t *fb)
{
  // logo
  fb_text14x16(fb->w / 2 + 20 - 80, fb->h / 2 + 20 - 150, "ap",
    10, "880", fb);
  fb_text14x16(fb->w / 2 + 10 - 80, fb->h / 2 + 10 - 150, "ap",
    10, "bb0", fb);
  fb_text14x16(fb->w / 2 - 80, fb->h / 2 - 150, "ap",
    10, "ff0", fb);
  fb_text14x16(fb->w / 2 + 20 + 50, fb->h / 2 + 20 - 170, "8",
    16, "803", fb);
  fb_text14x16(fb->w / 2 + 10 + 50, fb->h / 2 + 10 - 170, "8",
    16, "b05", fb);
  fb_text14x16(fb->w / 2 + 50, fb->h / 2 - 170, "8",
    16, "f07", fb);
  lcd_draw(lcd_mem_base, fb);
  parlcd_delay(2000);
  // ap8 name
  fb_text14x16(fb->w / 2, fb->h - 90, "actually perfect 8bit",
    3, "fff", fb);
  lcd_draw(lcd_mem_base, fb);
  parlcd_delay(2000);
  // enjoy text
  fb_fill(0, 0, fb->w, fb->h, "000", fb);
  fb_text14x16(fb->w / 2, fb->h / 2, "Enjoy! :)",
    4, "fff", fb);
  lcd_draw(lcd_mem_base, fb);
  parlcd_delay(1000);
  // grid with cursor
  fb_fill(0, 0, fb->w, fb->h, "000", fb);
  fb_block(0, 0, 1, "fff", fb);
  lcd_draw(lcd_mem_base, fb);
}

/* LCD color test */
void lcd_test(unsigned char *lcd_mem_base, fbuffer_t *fb)
{
  // orange
  fb_block(0, 0, 0, "f00", fb);
  fb_block(0, 1, 0, "e00", fb);
  fb_block(0, 2, 0, "d00", fb);
  fb_block(0, 3, 0, "b00", fb);
  fb_block(0, 4, 0, "900", fb);
  fb_block(0, 5, 0, "700", fb);
  fb_block(0, 6, 0, "500", fb);
  fb_block(0, 7, 0, "300", fb);
  fb_block(0, 8, 0, "100", fb);
  lcd_draw(lcd_mem_base, fb);
  parlcd_delay(100);

  // yellow
  fb_block(1, 0, 0, "ff0", fb);
  fb_block(1, 1, 0, "ee0", fb);
  fb_block(1, 2, 0, "dd0", fb);
  fb_block(1, 3, 0, "bb0", fb);
  fb_block(1, 4, 0, "990", fb);
  fb_block(1, 5, 0, "770", fb);
  fb_block(1, 6, 0, "550", fb);
  fb_block(1, 7, 0, "330", fb);
  fb_block(1, 8, 0, "110", fb);
  lcd_draw(lcd_mem_base, fb);
  parlcd_delay(100);

  // green
  fb_block(2, 0, 0, "0f0", fb);
  fb_block(2, 1, 0, "0e0", fb);
  fb_block(2, 2, 0, "0d0", fb);
  fb_block(2, 3, 0, "0b0", fb);
  fb_block(2, 4, 0, "090", fb);
  fb_block(2, 5, 0, "070", fb);
  fb_block(2, 6, 0, "050", fb);
  fb_block(2, 7, 0, "030", fb);
  fb_block(2, 8, 0, "010", fb);
  lcd_draw(lcd_mem_base, fb);
  parlcd_delay(100);

  // cyan
  fb_block(3, 0, 0, "0ff", fb);
  fb_block(3, 1, 0, "0ee", fb);
  fb_block(3, 2, 0, "0dd", fb);
  fb_block(3, 3, 0, "0bb", fb);
  fb_block(3, 4, 0, "099", fb);
  fb_block(3, 5, 0, "077", fb);
  fb_block(3, 6, 0, "055", fb);
  fb_block(3, 7, 0, "033", fb);
  fb_block(3, 8, 0, "011", fb);
  lcd_draw(lcd_mem_base, fb);
  parlcd_delay(100);

  // blue
  fb_block(4, 0, 0, "00f", fb);
  fb_block(4, 1, 0, "00e", fb);
  fb_block(4, 2, 0, "00d", fb);
  fb_block(4, 3, 0, "00b", fb);
  fb_block(4, 4, 0, "009", fb);
  fb_block(4, 5, 0, "007", fb);
  fb_block(4, 6, 0, "005", fb);
  fb_block(4, 7, 0, "003", fb);
  fb_block(4, 8, 0, "001", fb);
  lcd_draw(lcd_mem_base, fb);
  parlcd_delay(100);

  // magenta
  fb_block(5, 0, 0, "f0f", fb);
  fb_block(5, 1, 0, "e0e", fb);
  fb_block(5, 2, 0, "d0d", fb);
  fb_block(5, 3, 0, "b0b", fb);
  fb_block(5, 4, 0, "909", fb);
  fb_block(5, 5, 0, "707", fb);
  fb_block(5, 6, 0, "505", fb);
  fb_block(5, 7, 0, "303", fb);
  fb_block(5, 8, 0, "101", fb);
  lcd_draw(lcd_mem_base, fb);
}
