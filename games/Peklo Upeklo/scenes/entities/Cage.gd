extends Entity
class_name Cage
# Cage
# Cage interacts with tracks and torture instruments.


var grid_index:Vector2 = Vector2(-1, 0)
var direction:int = RIGHT
var playback_speed:float = 3.0
var should_end:bool = false
var should_collide:bool = false
var sins_num:int = 0
var id:int
var sins:Dictionary = {
	"Lust": false,
	"Gluttony": false,
	"Greed": false,
	"Sloth": false,
	"Wrath": false,
	"Envy": false,
	"Pride": false }

onready var tween:Tween = $Tween
onready var animation:AnimationPlayer = $AnimationPlayer
onready var orders = get_parent().get_parent().get_node("Orders")


# Set sins
func set_sins(arg:Array) -> void:
	for s in arg:
		sins[s] = true
		sins_num += 1
	update_sins()


# Remove sins
func remove_sins(arg:String) -> void:
	if sins[arg]:
		sins[arg] = false
		sins_num -= 1
		get_node("YSort/" + arg).hide()
		update_sins()
		# Record statistics
		orders.sin_punished(self, arg)
		orders.sin_punished(self, "Any")


# Update sins
func update_sins() -> void:
	if sins_num == 0:
		return
	var step:float = 2 * PI / sins_num
	var init:float = 0
	# Reorganize sins above cage
	for s in sins:
		if sins[s]:
			get_node("YSort/" + s).position = Vector2(10 * cos(init), 10 * sin(init))
			get_node("YSort/" + s).show()
			init += step


# Set playback speed of animations
func set_speed(speed:float) -> void:
	playback_speed = speed


# Move cage to new index in grid
func move_to(new_index:Vector2) -> void:
	# warning-ignore:return_value_discarded
	tween.interpolate_property(self, "global_position", global_position,
		get_parent().get_parent().grid_index_to_global_position(new_index),
		get_parent().spawn_time / 2, Tween.TRANS_LINEAR, Tween.EASE_IN_OUT)
	# warning-ignore:return_value_discarded
	tween.start()
	animation.playback_speed = playback_speed
	animation.play(get_direction_string())
	grid_index = new_index


# Get direction cage should move next from current tile
func get_direction_from_tile(tile) -> void:
	for i in range(4):
		if i != (self.direction + 2) % 4: # If direction is not opposite to current
			if tile.tracks[i]:
				direction = i
				return
	should_end = true


# Convert enum to string
func get_direction_string() -> String:
	match direction:
		RIGHT:
			return "Right"
		LEFT:
			return "Left"
		UP:
			return "Up"
		DOWN:
			return "Down"
	return ""


# Get index of new position
func next_postition() -> Vector2:
	match direction:
		RIGHT:
			return grid_index + Vector2.RIGHT
		LEFT:
			return grid_index + Vector2.LEFT
		UP:
			return grid_index + Vector2.UP
		DOWN:
			return grid_index + Vector2.DOWN
	return grid_index + Vector2.ZERO


# Signalled when a cage collides with another
func collision():
	match direction:
		RIGHT:
			global_position += Vector2.RIGHT * 28
			animation.play("Right Crash")
		LEFT:
			global_position += Vector2.LEFT * 28
			animation.play("Left Crash")
		UP:
			global_position += Vector2.UP * 28
			animation.play("Up Crash")
		DOWN:
			global_position += Vector2.DOWN * 28
			animation.play("Down Crash")
	orders.cage_collided()

# End a cage
func end():
	for key in sins.keys():
		if sins[key]:
			orders.sin_failed(self, key)
			orders.sin_failed(self, "Any")
	$Joint.hide()
	animation.playback_speed = 1.0
	animation.play("Fall")
	get_node("/root/Main/Level/MusicPlayer").dropped()
