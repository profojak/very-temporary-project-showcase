/********************************************************************
  Menu interface for MicroZed based MZ_APO board.
********************************************************************/

#define _POSIX_C_SOURCE 200112L

#include "menu.h"

/* 1 option menu init */
menu_t *menu1_init(char *text, char *rgb)
{
  menu_t *menu = (menu_t *)malloc(sizeof(menu_t));
  menu->rgb = rgb;
  menu->options = (char **)malloc(sizeof(char *));
  menu->options[0] = text;
  menu->size = 1;
  menu->active = 2;
  return menu;
}

/* 2 option menu init */
menu_t *menu2_init(char *text1, char *text2, char *rgb)
{
  menu_t *menu = (menu_t *)malloc(sizeof(menu_t));
  menu->rgb = rgb;
  menu->options = (char **)malloc(2 * sizeof(char *));
  menu->options[0] = text1;
  menu->options[1] = text2;
  menu->size = 2;
  menu->active = 0;
  return menu;
}
