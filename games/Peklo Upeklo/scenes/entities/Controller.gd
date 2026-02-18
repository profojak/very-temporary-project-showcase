extends Tile
class_name Controller
# Building controller entity
# Controller controls track connections and deletes tiles.


enum { NOT, OFF, ON }

var tile:Tile # which tile Controller controls
var logic:Array = [OFF, OFF, OFF, OFF]


# Show up on selected tile
func setup(index:Vector2, entity:Tile) -> void:
	grid_index = index
	tile = entity
	_draw_arrows()


# Update tracks between tiles
func update_connections(dir:int) -> void:
	if logic[(dir + 2) % 4] == NOT:
		var neighbors:Array = get_parent().get_neighbors(get_parent().global_position_to_grid_index(
				global_position))
		neighbors[(dir + 2) % 4].update_tracks(false, dir)
		tile.update_tracks(false, (dir + 2) % 4)
	elif logic[(dir + 2) % 4] == ON:
		var neighbors:Array = get_parent().get_neighbors(get_parent().global_position_to_grid_index(
				global_position))
		neighbors[(dir + 2) % 4].update_tracks(true, dir)
		tile.update_tracks(true, (dir + 2) % 4)
	_draw_arrows()


# Draw track connection arrows
func _draw_arrows() -> void:
	var tiles:Array = tile.get_connectable_neighbors()
	var counter:int = 0
	for t in tiles:
		if tile.tracks[counter] == true and t != INVALID:
			sprites[counter].texture = load("res://assets/entities/controller/Not.png")
			logic[counter] = NOT
			sprites[counter].show()
		else:
			if t == INVALID:
				sprites[counter].hide()
			elif t == VALID:
				if tile.tracks_cur < tile.tracks_max:
					sprites[counter].texture = load("res://assets/entities/controller/On.png")
					logic[counter] = ON
					sprites[counter].show()
				else:
					sprites[counter].texture = load("res://assets/entities/controller/Off.png")
					logic[counter] = OFF
					sprites[counter].show()
			elif t == FULL or t == EMPTY:
				sprites[counter].texture = load("res://assets/entities/controller/Off.png")
				logic[counter] = OFF
				sprites[counter].show()
		counter = counter + 1
