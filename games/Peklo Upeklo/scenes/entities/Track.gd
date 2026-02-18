extends Tile
class_name Track
# Track in Level grid
# Track connects Torture entities and lets cages pass.

var has_junction:bool = false
var junction_dir:int = RIGHT

onready var junction:Sprite = $Junction/Arrow


# Show junction when navigation phase starts
func show_junction() -> void:
	var num_of_tracks:int = 0
	for b in tracks:
		if b == true:
			num_of_tracks += 1
	if num_of_tracks > 2:
		$Junction.show()
		has_junction = true
		update_junction()


# Update direction of junction
func update_junction() -> void:
	match junction_dir:
		UP:
			junction_dir = RIGHT
			junction.rotation_degrees = 0
		RIGHT:
			junction_dir = DOWN
			junction.rotation_degrees = 90
		DOWN:
			junction_dir = LEFT
			junction.rotation_degrees = 180
			if grid_index == Vector2(0, 0):
				update_junction()
		LEFT:
			junction_dir = UP
			junction.rotation_degrees = 270
	if not tracks[junction_dir]:
		update_junction()


# Try to switch a junction if exists
func try_switch(dir):
	if has_junction:
		junction_dir = ((80 + dir - 1) % 4)
		update_junction()
