/********************************************************************
  LCD utilities for MicroZed based MZ_APO board.
********************************************************************/

#ifndef LCD_H
#define LCD_H

#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdbool.h>

#include "../mzapo/mzapo_parlcd.h"
#include "../gui/fbuffer.h"

void lcd_init(unsigned char *lcd_mem_base, unsigned int size);

void lcd_draw(unsigned char *lcd_mem_base, fbuffer_t *fb);

void lcd_splash(unsigned char *lcd_mem_base, fbuffer_t *fb);

void lcd_test(unsigned char *lcd_mem_base, fbuffer_t *fb);

#endif /*LCD_H*/
