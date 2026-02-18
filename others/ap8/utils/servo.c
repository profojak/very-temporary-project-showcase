/********************************************************************
  Servo utilities for MicroZed based MZ_APO board.
********************************************************************/

#define _POSIX_C_SOURCE 200112L

#include "servo.h"

static const int PWM[3][3] = {
  { 100000, 100000, 70000 },
  { 160000, 160000, 120000 },
  { 220000, 220000, 170000 }
};

/* servo set */
void servo_set(unsigned char *servo_mem_base)
{
  *(volatile uint32_t *)(servo_mem_base + SERVOPS2_REG_CR_o) = 0x10f;
  *(volatile uint32_t *)(servo_mem_base + SERVOPS2_REG_PWMPER_o) =
    1000000;
  *(volatile uint32_t *)(servo_mem_base + SERVOPS2_REG_PWM1_o) =
    PWM[2][0];
  parlcd_delay(500);
  *(volatile uint32_t *)(servo_mem_base + SERVOPS2_REG_PWM2_o) =
    PWM[2][1];
  parlcd_delay(500);
  *(volatile uint32_t *)(servo_mem_base + SERVOPS2_REG_PWM3_o) =
    PWM[2][2];
}

/* turn the knob */
void servo_move(unsigned char *led_mem_base, short num, char color)
{
  if (num < 0 || num > 2) { return; }
  switch (color) {
  case 'r':
    *(volatile uint32_t *)(led_mem_base + SERVOPS2_REG_PWM1_o) =
      PWM[num][0];
    break;
  case 'g':
    *(volatile uint32_t *)(led_mem_base + SERVOPS2_REG_PWM2_o) =
      PWM[num][1];
    break;
  case 'b':
    *(volatile uint32_t *)(led_mem_base + SERVOPS2_REG_PWM3_o) =
      PWM[num][2];
    break;
  }
}

/* get knob value */
uint32_t knob_val(unsigned char *led_mem_base, char color)
{
  uint32_t value = *(volatile uint32_t *)(led_mem_base +
    SPILED_REG_KNOBS_8BIT_o), mask = 0x000000ff;
  switch (color) {
  case 'b':
    value = value & mask;
    if (value < 100) { return 2;
    } else if (value < 235) { return 0;
    } else { return 1; }
    break;
  case 'g':
    value = (value & (mask << 8)) >> 8;
    if (value < 15) { return 0;
    } else if (value > 40) { return 2;
    } else { return 1; }
    break;
  case 'r':
    value = (value & (mask << 16)) >> 16;
    break;
  }
  return -1;
}

/* servo test */
void servo_test(unsigned char *servo_mem_base)
{
  *(volatile uint32_t *)(servo_mem_base + SERVOPS2_REG_CR_o) = 0x10f;
  *(volatile uint32_t *)(servo_mem_base + SERVOPS2_REG_PWMPER_o) =
    1000000;
  *(volatile uint32_t *)(servo_mem_base + SERVOPS2_REG_PWM1_o) =
    PWM[0][0];
  parlcd_delay(500);
  *(volatile uint32_t *)(servo_mem_base + SERVOPS2_REG_PWM2_o) =
    PWM[0][1];
  parlcd_delay(500);
  *(volatile uint32_t *)(servo_mem_base + SERVOPS2_REG_PWM3_o) =
    PWM[0][2];
  parlcd_delay(500);
  *(volatile uint32_t *)(servo_mem_base + SERVOPS2_REG_PWM1_o) =
    PWM[1][0];
  parlcd_delay(500);
  *(volatile uint32_t *)(servo_mem_base + SERVOPS2_REG_PWM2_o) =
    PWM[1][1];
  parlcd_delay(500);
  *(volatile uint32_t *)(servo_mem_base + SERVOPS2_REG_PWM3_o) =
    PWM[1][2];
  parlcd_delay(500);
  *(volatile uint32_t *)(servo_mem_base + SERVOPS2_REG_PWM1_o) =
    PWM[2][0];
  parlcd_delay(500);
  *(volatile uint32_t *)(servo_mem_base + SERVOPS2_REG_PWM2_o) =
    PWM[2][1];
  parlcd_delay(500);
  *(volatile uint32_t *)(servo_mem_base + SERVOPS2_REG_PWM3_o) =
    PWM[2][2];
}
