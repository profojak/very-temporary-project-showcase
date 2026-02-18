/********************************************************************
  LED utilities for MicroZed based MZ_APO board.
********************************************************************/

#ifndef LED_H
#define LED_H

#include <stdlib.h>
#include <stdint.h>

#include "../mzapo/mzapo_regs.h"
#include "../mzapo/mzapo_parlcd.h"

void led_line_set(unsigned char *led_mem_base, int leds);

void led_rgb_set(unsigned char *led_mem_base, char *rgb);

void led_line_test(unsigned char *led_mem_base);

void led_rgb_test(unsigned char *led_mem_base);

#endif /*LED_H*/
