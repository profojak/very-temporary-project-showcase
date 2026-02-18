/********************************************************************
  Menu interface for MicroZed based MZ_APO board.
********************************************************************/

#ifndef MENU_H
#define MENU_H

#include <stdlib.h>
#include <stdint.h>
#include <string.h>

typedef struct {
  char **options; /* array with text of options */
  char *rgb;      /* rgb */
  short size;     /* size */
  short active;   /* active option (knob value for 1-option menu) */
} menu_t;

menu_t *menu2_init(char *text1, char *text2, char *rgb);

menu_t *menu1_init(char *text, char *rgb);

#endif /*MENU_H*/
