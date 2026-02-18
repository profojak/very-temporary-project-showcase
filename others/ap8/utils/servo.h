/********************************************************************
  Servo utilities for MicroZed based MZ_APO board.
********************************************************************/

#ifndef SERVO_H
#define SERVO_H

#include <stdlib.h>
#include <stdint.h>

#include "../mzapo/mzapo_parlcd.h"
#include "../mzapo/mzapo_regs.h"

void servo_set(unsigned char *servo_mem_base);

void servo_move(unsigned char *servo_mem_base, short num, char color);

uint32_t knob_val(unsigned char *led_mem_base, char color);

void servo_test(unsigned char *servo_mem_base);

#endif /*SERVO_H*/
