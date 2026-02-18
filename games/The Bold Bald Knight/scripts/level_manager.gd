extends Node


var objectives: Array = [
	"Save the princess!",
	"Save the princess as a bold knight in a shiny armor!",
	"Save the princess from the monsters!",
	"Meet the princess after the beauty procedure!",
	"Find and bring a golden comb to the princess!",
	"Run to the princess without getting hit!",
	"Entertain the princess by forcing one skeleton to hit another!",
	"Find and bring again the golden comb to the princess!",
]


var cutscene: bool = false
var level: int = 7


var level_2 = [false, 0]
var level_4 = false
var level_5 = true
var level_6 = false
var level_7 = false


func level_2_update():
	level_2[1] += 1
	if level_2[1] == 4:
		level_2[0] = true


func next_day():
	cutscene = false
	get_tree().reload_current_scene()
