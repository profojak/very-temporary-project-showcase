extends Entity
class_name Tile
# Tile entity
# Tile is any scene player builds that appear on Level grid.


enum { INVALID, EMPTY, FULL, VALID }

var grid_index:Vector2
var tracks_cur:int = 0
var enabled:bool = true
var tracks:Array = [false, false, false, false]
export var type:String = "Track"
onready var sprites:Array = [$Parent/Up, $Parent/Right, $Parent/Down, $Parent/Left]

export var tracks_max:int = 0


# Get logic array of connectable tiles
func get_connectable_neighbors() -> Array:
	var connections:Array = []
	for n in get_parent().get_neighbors(grid_index):
		if n:
			if n.tracks_cur < n.tracks_max:
				connections.append(VALID)
			else:
				connections.append(FULL)
		elif n == null:
			connections.append(EMPTY)
		elif n == false:
			connections.append(INVALID)
	return connections


# Create tracks on tile placement
func create_tracks() -> void:
	if grid_index == Vector2(0, 0):
		if tracks_cur < tracks_max:
			update_tracks(true, LEFT)
	var counter:int = 0
	for n in get_parent().get_neighbors(grid_index):
		if n:
			if tracks_cur < tracks_max and n.tracks_cur < n.tracks_max:
				update_tracks(true, counter)
				n.update_tracks(true, (counter + 2) % 4)
		counter = counter + 1


# Remove tracks connecting this tile
func remove_tracks() -> void:
	var counter:int = 0
	for n in get_parent().get_neighbors(grid_index):
		if n:
			n.update_tracks(false, (counter + 2) % 4)
		counter = counter + 1


# Update tracks in specific direction
func update_tracks(create:bool, dir:int) -> void:
	if create:
		tracks_cur = tracks_cur + 1
		sprites[dir].show()
		tracks[dir] = true
	else:
		if tracks[dir]:
			tracks_cur = tracks_cur - 1
			sprites[dir].hide()
			tracks[dir] = false


# Disable tile
func disable() -> void:
	enabled=false


# Return tile type
func get_type() -> String:
	return type
