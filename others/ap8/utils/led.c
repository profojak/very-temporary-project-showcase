/********************************************************************
  LED utilities for MicroZed based MZ_APO board.
********************************************************************/

#define _POSIX_C_SOURCE 200112L

#include "led.h"

/* set LEDs on 32 LED line */
void led_line_set(unsigned char *led_mem_base, int leds)
{
  int i;
  unsigned int mask = 1;
  uint32_t line = 0, set = 0x0000003e;
  for (i = 0; i < 6; i++) {
    if (leds & (mask << i)) {
      line += set << (5 * i);
    }
  }
  *(volatile uint32_t *)(led_mem_base + SPILED_REG_LED_LINE_o) = line;
}

/* color LEDs set */
void led_rgb_set(unsigned char *led_mem_base, char *rgb)
{
  char r, g, b;
  uint32_t c;
  r = rgb[0] > 57 ? rgb[0] - 88 : rgb[0] - 48;
  g = rgb[1] > 57 ? rgb[1] - 88 : rgb[1] - 48;
  b = rgb[2] > 57 ? rgb[2] - 88 : rgb[2] - 48;
  c = 0x0 + ((r * 16) << 16) + ((g * 16) << 8) + (b * 16);
  *(volatile uint32_t *)(led_mem_base + SPILED_REG_LED_RGB1_o) = c;
  *(volatile uint32_t *)(led_mem_base + SPILED_REG_LED_RGB2_o) = c;
}

/* test 32 LED line */
void led_line_test(unsigned char *led_mem_base) {
  int i;
  uint32_t line = 1;
  for (i = 0; i < 44; i++) {
    line = (line << 1);
    if (i < 6) {
      line++;
    }  
    *(volatile uint32_t *)(led_mem_base + SPILED_REG_LED_LINE_o) = line;
    parlcd_delay(30);
  }

  parlcd_delay(200);
  led_line_set(led_mem_base, 0x0);
  parlcd_delay(200);
  led_line_set(led_mem_base, 0x1);
  parlcd_delay(200);
  led_line_set(led_mem_base, 0x2);
  parlcd_delay(200);
  led_line_set(led_mem_base, 0x4);
  parlcd_delay(200);
  led_line_set(led_mem_base, 0x8);
  parlcd_delay(200);
  led_line_set(led_mem_base, 0x10);
  parlcd_delay(200);
  led_line_set(led_mem_base, 0x20);
  parlcd_delay(200);
  led_line_set(led_mem_base, 0x1 + 0x4);
  parlcd_delay(200);
  led_line_set(led_mem_base, 0x2 + 0x8);
  parlcd_delay(200);
  led_line_set(led_mem_base, 0x4 + 0x10);
  parlcd_delay(200);
  led_line_set(led_mem_base, 0x8 + 0x20);
  parlcd_delay(200);
  led_line_set(led_mem_base, 0x1 + 0x4 + 0x10);
  parlcd_delay(200);
  led_line_set(led_mem_base, 0x2 + 0x8 + 0x20);
  parlcd_delay(200);
  led_line_set(led_mem_base, 0x1 + 0x2 + 0x4 + 0x8 + 0x10 + 0x20);
  parlcd_delay(200);
  led_line_set(led_mem_base, 0x0);
}

/* test color LEDs */
void led_rgb_test(unsigned char *led_mem_base)
{
  int i;
  char rgb[3] = "f00";

  // red
  for (i = 0; i < 15; i++) {
    led_rgb_set(led_mem_base, rgb);
    parlcd_delay(40);
    rgb[0]--;
  }
  parlcd_delay(300);

  // green
  rgb[0] = '0';
  rgb[1] = 'f';
  for (i = 0; i < 15; i++) {
    led_rgb_set(led_mem_base, rgb);
    parlcd_delay(40);
    rgb[1]--;
  }
  parlcd_delay(300);

  // blue
  rgb[1] = '0';
  rgb[2] = 'f';
  for (i = 0; i < 15; i++) {
    led_rgb_set(led_mem_base, rgb);
    parlcd_delay(40);
    rgb[2]--;
  }
}
