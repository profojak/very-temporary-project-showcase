/********************************************************************
  ap8 - actually perfect 8bit music sequencer

  This is semestral work for B0B35APO course: Computer Architecture
  at FEE CTU.

  Made by Jakub Profota and Jenda Kucera in 2020 summer semester.
********************************************************************/

#define _POSIX_C_SOURCE 200112L
#define WIDTH 480
#define HEIGHT 320
#define BLKW 6
#define BLKH 10

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <unistd.h>

#include "mzapo/mzapo_phys.h"
#include "mzapo/mzapo_parlcd.h"
#include "mzapo/mzapo_regs.h"
#include "utils/lcd.h"
#include "utils/led.h"
#include "utils/servo.h"
#include "gui/fbuffer.h"
#include "gui/menu.h"

/* input enum */
enum input_enum {
  LEFT, RIGHT, UP, DOWN, MINUS, PLUS, ESC, S, P, X, B
};

/* app state enum */
enum state_enum {
  SPLASH, CREATE, EXIT, PLAY, SPEED, BACKLIGHT
};

/* play step enum */
enum play_enum {
  START = 0, FB_UP = 8
};

/* color enum */
enum color_enum {
  RED = 0, YELLOW = 1, GREEN = 2, CYAN = 3, BLUE = 4, MAGENTA = 5,
  WHITE = 6, BLACK = 7
};

/* color array */
char COLOR[8][3] = {
  "f00", "ff0", "0f0", "0ff", "00f", "f0f", "fff", "000" };

/* process input */
int get_input(char input)
{
  switch (input) {
  case 'l': case 'L': return LEFT;
  case 'r': case 'R': return RIGHT;
  case 'u': case 'U': return UP;
  case 'd': case 'D': return DOWN;
  case 'x': case 'X': return X;
  case 'e': case 'E': return ESC;
  case 's': case 'S': return S;
  case 'p': case 'P': return P;
  case 'b': case 'B': return B;
  case '-': return MINUS;
  case '+': return PLUS;
  default: return -1;
  }
}

/* grid realloc */
char *grid_realloc(char *grid, int grid_size)
{
  char *temp = NULL;
  while (temp == NULL) {
    temp = realloc(grid, grid_size);
  }
  memset(temp + grid_size - BLKW * BLKH, 0, BLKW * BLKH);
  return temp;
}

/* block from index to frame buffer */
int set_block(char *grid, int index, int grid_page, fbuffer_t *fb)
{
  if (grid[index + grid_page * BLKW * BLKH]) { 
    fb_block(index % BLKW, index / BLKW, 0, COLOR[index % BLKW], fb);
    return index / BLKW + grid_page * BLKH;
  } else {
    fb_block(index % BLKW, index / BLKW, 0, COLOR[BLACK], fb);
  }
  return 0;
}

/* grid of blocks to frame buffer */
void set_grid(char *grid, int grid_page, fbuffer_t *fb)
{
  int i;
  for (i = 0; i < BLKW * BLKH; i++) {
    set_block(grid, i, grid_page, fb);
  }
}

/* set intensity of colors */
void set_light(char c)
{
  int i, j;
  c += c < 10 ? '0' : 'a' - 10;
  for (i = 0; i < 8; i++) {
    for (j = 0; j < 3; j++) {
      if (COLOR[i][j] != '0') {
        COLOR[i][j] = c;
      }
    }
  }
}

/* close menu */
void menu_close(menu_t *menu, char *grid, int grid_page,
  int grid_active, int *STATE, fbuffer_t *fb)
{
  free(menu);
  menu = NULL;
  fb_fill(0, 0, fb->w, fb->h, COLOR[BLACK], fb);
  set_grid(grid, grid_page, fb);
  fb_block(grid_active % BLKW, grid_active / BLKW, 1,
    COLOR[WHITE], fb);
  *STATE = CREATE;
}

/* increase active grid index */
void index_increase(char *grid, int *grid_active, int *grid_page,
  int val, fbuffer_t *fb)
{
  set_block(grid, *grid_active, *grid_page, fb);
  (*grid_active) -= val;
  if ((*grid_active) < 0) {
    (*grid_active) += BLKW * BLKH;
    (*grid_page)--;
    if ((*grid_page) < 0) {
      *grid_page = 0;
    }
    set_grid(grid, *grid_page, fb);
  }
  fb_block((*grid_active) % BLKW, (*grid_active) / BLKW, 1,
    COLOR[WHITE], fb);
}

/* decrease active grid index */
int index_decrease(char *grid, int *grid_active, int *grid_page,
  int grid_size, int val, fbuffer_t *fb)
{
  int ret = 0;
  set_block(grid, *grid_active, *grid_page, fb);
  (*grid_active) += val;
  if ((*grid_active) >= BLKW * BLKH) {
    (*grid_active) = (*grid_active) % (BLKW * BLKH);
    (*grid_page)++;
    if (((*grid_page) + 1) * BLKW * BLKH == grid_size) {
      ret = 1;
    }
    set_grid(grid, *grid_page, fb);
  }
  fb_block((*grid_active) % BLKW, (*grid_active) / BLKW, 1,
    COLOR[WHITE], fb);
  return ret;
}

/* play created song */
void play(unsigned char *led_mem_base, unsigned char *lcd_mem_base,
  char *grid, int low, int speed, fbuffer_t *fb)
{
  int i, j, k, color, line;
  fb_fill(0, 0, fb->w, fb->h, COLOR[BLACK], fb);
  set_grid(grid, 0, fb);
  for (i = 0; i <= low; i++) {
    for (j = 0; j < 32; j++) {
      if (j == START) {
        color = 7;
        line = 0;
        for (k = 0; k < BLKW; k++) {
          // draw blocks
          if (grid[(i + BLKH) * BLKW + k] == 1) {
            fb_block(k, 10, 0, COLOR[k], fb);
          } else {
            fb_block(k, 10, 0, COLOR[BLACK], fb);
          }

          // set color LEDs and 32 LED line
          if (grid[i * BLKW + k] == 1) {
            line += (1 << (5 - k));
            if (color != 7) {
              color = 6;
            } else {
              color = k;
            }
          }
        }
        led_line_set(led_mem_base, line);
        led_rgb_set(led_mem_base, COLOR[color]);
      }

      // move frame buffer up on every 8th iteration
      if (j % FB_UP == 0) {
        fb_up(fb);
        lcd_draw(lcd_mem_base, fb);
      }

      parlcd_delay(speed * 10);
    }
  }
  led_rgb_set(led_mem_base, COLOR[BLACK]);
  led_line_set(led_mem_base, 0);
}

/* check if all mzapo kit components work */
void test(unsigned char *led_mem_base, unsigned char *lcd_mem_base,
  unsigned char *servo_mem_base, fbuffer_t *fb)
{
  lcd_test(lcd_mem_base, fb);
  led_line_test(led_mem_base);
  led_rgb_test(led_mem_base);
  servo_test(servo_mem_base);
}

/* main */
int main(int argc, char *argv[])
{
  unsigned char *led_mem_base = (unsigned char *)map_phys_address(
    SPILED_REG_BASE_PHYS, SPILED_REG_SIZE, 0);
  unsigned char *lcd_mem_base = (unsigned char *)map_phys_address(
    PARLCD_REG_BASE_PHYS, PARLCD_REG_SIZE, 0);
  unsigned char *servo_mem_base = (unsigned char *)map_phys_address(
    SERVOPS2_REG_BASE_PHYS, SERVOPS2_REG_SIZE, 0);
  //unsigned char *audio_mem_base = (unsigned char *)map_phys_address(
    //AUDIOPWM_REG_BASE_PHYS, AUDIOPWM_REG_SIZE, 0);

  // init device
  lcd_init(lcd_mem_base, WIDTH * HEIGHT);
  led_line_set(led_mem_base, 0x0);
  led_rgb_set(led_mem_base, COLOR[BLACK]);
  servo_set(servo_mem_base);
  fbuffer_t *fb = fb_init(WIDTH, HEIGHT);

  //test(led_mem_base, lcd_mem_base, servo_mem_base, fb);
  //return 0;

  // init variables
  int grid_size = 2 * BLKW * BLKH, grid_active = 0, grid_page = 0,
    low = 0, ret, speed, RUN = 1, STATE = CREATE;
  char *grid = malloc(grid_size), input;
  memset(grid, 0, grid_size);
  menu_t *menu = NULL;
  speed = 2 - knob_val(led_mem_base, 'g');

  lcd_splash(lcd_mem_base, fb);

  // main loop
  while (RUN) {
    input = getchar();
    if (get_input(input) == -1) { continue; }

    switch (STATE) {
    // creating song
    case CREATE:
      switch (get_input(input)) {
      case RIGHT: // move cursor to the right
        if (index_decrease(grid, &grid_active, &grid_page,
          grid_size, 1, fb)) {
          grid_size += BLKW * BLKH;
          grid = grid_realloc(grid, grid_size);
        }
        break;
      case LEFT: // move cursor to the left
        index_increase(grid, &grid_active, &grid_page, 1, fb);
        break;
      case UP: // move cursor up
        index_increase(grid, &grid_active, &grid_page, 6, fb);
        break;
      case DOWN: // move cursor down
        if (index_decrease(grid, &grid_active, &grid_page,
          grid_size, 6, fb)) {
          grid_size += BLKW * BLKH;
          grid = grid_realloc(grid, grid_size);
        }
        break;
      case X: // place block
        if(grid[grid_active + grid_page * BLKW * BLKH]) {
          grid[grid_active + grid_page * BLKW * BLKH] = 0;
        } else {
          grid[grid_active + grid_page * BLKW * BLKH] = 1;
        }
        ret = set_block(grid, grid_active, grid_page, fb);
        if (ret > low) { low = ret; }
        fb_block(grid_active % BLKW, grid_active / BLKW, 1,
          COLOR[WHITE], fb);
        break;
      case ESC: // open exit menu
        menu = menu2_init("Cancel", "Exit", COLOR[RED]);
        fb_menu2(menu, fb);
        STATE = EXIT;
        break;
      case B: // open brightness menu
        menu = menu1_init("Backlight", COLOR[BLUE]);
        menu->active = knob_val(led_mem_base, 'b');
        fb_menu1(menu, fb);
        STATE = BACKLIGHT;
        break;
      case S: // open speed menu
        menu = menu1_init("Speed", COLOR[GREEN]);
        menu->active = knob_val(led_mem_base, 'g');
        fb_menu1(menu, fb);
        STATE = SPEED;
        break;
      case P: // open play menu
        menu = menu2_init("Cancel", "Play", COLOR[WHITE]);
        fb_menu2(menu, fb);
        STATE = PLAY;
        break;
      default:
        continue;
      }
      break;
    // exit menu
    case EXIT:
      switch (get_input(input)) {
      case LEFT: // previous option
        menu->active = (menu->active - 1 + menu->size) % menu->size;
        fb_menu2(menu, fb);
        break;
      case RIGHT: // next option
        menu->active = (menu->active + 1) % menu->size;
        fb_menu2(menu, fb);
        break;
      case UP: // previous option
        menu->active = (menu->active - 1 + menu->size) % menu->size;
        fb_menu2(menu, fb);
        break;
      case DOWN: // next option
        menu->active = (menu->active + 1) % menu->size;
        fb_menu2(menu, fb);
        break;
      case X: // select
        if (menu->active == 0) {
          menu_close(menu, grid, grid_page, grid_active, &STATE, fb);
          break;
        } else {
          free(menu);
          menu = NULL;
          RUN = 0;
        }
        break;
      case ESC: // close menu
        menu_close(menu, grid, grid_page, grid_active, &STATE, fb);
        break;
      default:
        continue;
      }
      break;
    // backlight menu
    case BACKLIGHT:
      switch (get_input(input)) {
      case LEFT: case MINUS: // less
        servo_move(servo_mem_base, menu->active - 1, 'b');
        parlcd_delay(300);
        menu->active = knob_val(led_mem_base, 'b');
        set_light(menu->active * 5 + 5);
        fb_menu1(menu, fb);
        break;
      case RIGHT: case PLUS: // more
        servo_move(servo_mem_base, menu->active + 1, 'b');
        parlcd_delay(300);
        menu->active = knob_val(led_mem_base, 'b');
        set_light(menu->active * 5 + 5);
        fb_menu1(menu, fb);
        break;
      case ESC: // close menu
        menu_close(menu, grid, grid_page, grid_active, &STATE, fb);
        break;
      default:
        continue;
      }
      break;
    // speed menu
    case SPEED:
      switch (get_input(input)) {
      case LEFT: case MINUS: // less
        servo_move(servo_mem_base, menu->active - 1, 'g');
        parlcd_delay(300);
        menu->active = knob_val(led_mem_base, 'g');
        speed = 2 - menu->active;
        fb_menu1(menu, fb);
        break;
      case RIGHT: case PLUS: // more
        servo_move(servo_mem_base, menu->active + 1, 'g');
        parlcd_delay(300);
        menu->active = knob_val(led_mem_base, 'g');
        speed = 2 - menu->active;
        fb_menu1(menu, fb);
        break;
      case ESC: // close menu
        menu_close(menu, grid, grid_page, grid_active, &STATE, fb);
        break;
      default:
        continue;
      }
      break;
    // play menu
    case PLAY:
      switch (get_input(input)) {
      case LEFT: // previous option
        menu->active = (menu->active - 1 + menu->size) % menu->size;
        fb_menu2(menu, fb);
        break;
      case RIGHT: // next option
        menu->active = (menu->active + 1) % menu->size;
        fb_menu2(menu, fb);
        break;
      case UP: // previous option
        menu->active = (menu->active - 1 + menu->size) % menu->size;
        fb_menu2(menu, fb);
        break;
      case DOWN: // next option
        menu->active = (menu->active + 1) % menu->size;
        fb_menu2(menu, fb);
        break;
      case X: // select
        if (menu->active == 0) {
          menu_close(menu, grid, grid_page, grid_active, &STATE, fb);
          break;
        } else {
          play(led_mem_base, lcd_mem_base, grid, low, speed, fb);
          menu_close(menu, grid, grid_page, grid_active, &STATE, fb);
        }
        break;
      case ESC: // close menu
        menu_close(menu, grid, grid_page, grid_active, &STATE, fb);
        break;
      default:
        continue;
      }
      break;
    }
    lcd_draw(lcd_mem_base, fb);
  }

  // <3 screen and cleaning
  led_line_set(led_mem_base, 0x0);
  led_rgb_set(led_mem_base, COLOR[BLACK]);
  fb_fill(0, 0, fb->w, fb->h, COLOR[BLACK], fb);
  fb_text14x16(fb->w / 2, fb->h / 2, "<3", 10, "f00", fb);
  lcd_draw(lcd_mem_base, fb);
   
  free(grid);
  free(fb->r);
  free(fb->g);
  free(fb->b);
  free(fb);

  return 0;
}
