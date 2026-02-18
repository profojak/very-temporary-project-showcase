extends Node2D
class_name Main
# Main
# Main connects Level, Selection, and View logic and parses level configuration.

var track:PackedScene = load("res://scenes/entities/Track.tscn")
var torture:PackedScene = load("res://scenes/entities/Torture.tscn")
var item:PackedScene = load("res://scenes/entities/Item.tscn")
var level_phase:int = 1 # 1 - building; 2 - playing

var faces = [load("res://assets/view/Face1.png"), load("res://assets/view/Face2.png"),
			load("res://assets/view/Face3.png"), load("res://assets/view/Face4.png")]
var rights = [load("res://assets/view/Right1.png"), load("res://assets/view/Right2.png")]
var lefts = [load("res://assets/view/Left1.png"), load("res://assets/view/Left2.png")]
var bodies = [load("res://assets/view/Body1.png"), load("res://assets/view/Body2.png"),
			load("res://assets/view/Body3.png")]

onready var instructions:Instructions = $Instructions
onready var level:Grid = $Level
onready var selection:Grid = $Selection
onready var simulation:Simulation = $Level/Simulation

var level_names = [
	"The Sinferno",
	"The Eternal Buffering",
	"The In-Law Visit",
	"The Eternal Hangover",
	"The Group Project",
	"The Deadline",
	"The Infinite Hellyeah"
]


# Get level number
func get_level_number():
	return get_node("/root/Statistics").level_number


# Set level number
func set_level_number(number: int):
	get_node("/root/Statistics").level_number = number


func _ready():
	$LevelLabelContainer/Label/AnimationPlayer.play("fade")
	$LevelLabelContainer/Label.text = "Level " + str(get_node("/root/Statistics").level_number + 2) + ":\n" + level_names[get_node("/root/Statistics").level_number + 1]
	print_debug("Starting level", get_level_number())
	_parse_text_file()
	selection.on_click(Vector2(0, 0))


# Process input (very complex)
func _input(event: InputEvent):
	if $Menu.visible:
		return
	if event is InputEventMouseButton and event.pressed and event.button_index == BUTTON_LEFT:
		if instructions.opened and instructions.is_close_clicked(get_global_mouse_position()):
			instructions.close(get_viewport_rect())
			return
		elif !instructions.opened and instructions.is_open_clicked(get_global_mouse_position()):
			instructions.open()
			return
	if !instructions.opened:
		if event is InputEventMouseButton and event.pressed and event.button_index == BUTTON_LEFT:
			if level_phase == 1:
				if event is InputEventMouseButton and event.pressed and event.button_index == BUTTON_LEFT:
					var level_index:Vector2 = level.global_position_to_grid_index(get_global_mouse_position())
					var selection_index:Vector2 = selection.global_position_to_grid_index(get_global_mouse_position())
					if level.is_in_grid(level_index):
						if selection.get_item().type == "Track":
							var entity:Track = track.instance()
							level.on_click(level_index, entity, 1)
						else:
							var entity:Torture = torture.instance()
							entity.setup(selection.get_item().type)
							level.on_click(level_index, entity, 1)
					elif selection.is_in_grid(selection_index):
						selection.on_click(selection_index)
			elif level_phase == 2:
				if event is InputEventMouseButton and event.pressed and event.button_index == BUTTON_LEFT:
					var level_index:Vector2 = level.global_position_to_grid_index(get_global_mouse_position())
					if level.is_in_grid(level_index):
						level.on_click(level_index, null, 2)
	if event.is_action("down"):
		var grid_pos = level.global_position_to_grid_index(get_global_mouse_position())
		var tiles = $Level.tiles
		if grid_pos.x >= 0 and grid_pos.y >= 0 and grid_pos.x < len(tiles) and grid_pos.y < len(tiles[0]):
			if tiles[grid_pos.x][grid_pos.y] is Track:
				tiles[grid_pos.x][grid_pos.y].try_switch(Entity.DOWN)
	if event.is_action("right"):
		var grid_pos = level.global_position_to_grid_index(get_global_mouse_position())
		var tiles = $Level.tiles
		if grid_pos.x >= 0 and grid_pos.y >= 0 and grid_pos.x < len(tiles) and grid_pos.y < len(tiles[0]):
			if tiles[grid_pos.x][grid_pos.y] is Track:
				tiles[grid_pos.x][grid_pos.y].try_switch(Entity.RIGHT)
	if event.is_action("left"):
		var grid_pos = level.global_position_to_grid_index(get_global_mouse_position())
		var tiles = $Level.tiles
		if grid_pos.x >= 0 and grid_pos.y >= 0 and grid_pos.x < len(tiles) and grid_pos.y < len(tiles[0]):
			if tiles[grid_pos.x][grid_pos.y] is Track:
				tiles[grid_pos.x][grid_pos.y].try_switch(Entity.LEFT)
	if event.is_action("up"):
		var grid_pos = level.global_position_to_grid_index(get_global_mouse_position())
		var tiles = $Level.tiles
		if grid_pos.x >= 0 and grid_pos.y >= 0 and grid_pos.x < len(tiles) and grid_pos.y < len(tiles[0]):
			if tiles[grid_pos.x][grid_pos.y] is Track:
				tiles[grid_pos.x][grid_pos.y].try_switch(Entity.UP)


# Switch phase from building to navigation phase
func switch_phase(phase:int) -> void:
	if phase == 1:
		pass
	elif phase == 2:
		level_phase = phase
		selection.hide()
		$Track.show()
		play_animation()
		$Peklo.show()
		$Instructions.hide()
		level.switch_phase()


func play_animation():
	$Track/AnimationPlayer.play("Loop")


# Change look of sinner in cage
func change_sinner():
	$Background/Cage/Face.texture = faces[randi() % 4]
	$Background/Cage/Right.texture = rights[randi() % 2]
	$Background/Cage/Left.texture = lefts[randi() % 2]
	$Background/Cage/Body.texture = bodies[randi() % 3]


# Reduce count of stamp
func reduce_count(item_type:String) -> bool:
	return selection.reduce_selected_item_count(item_type)


# Increase count of stamp
func increase_count(item_type:String) -> void:
	selection.increase_selected_item_count(item_type)


# Resume game
func resume() -> void:
	if level_phase==2:
		$Level/Timer.start()


# Pause game
func pause() -> void:
	$Level/Timer.stop()


# Restart game
func restart() -> void:
	# warning-ignore:return_value_discarded
	get_tree().reload_current_scene()


# Generate list of cages from configuration
func _generate_cages_from_settings(cages:Array) -> Array:
	randomize()
	var prepared_cages:Array =[]
	while !cages.empty():
		var i:int = randi() % cages.size()
		prepared_cages.append(cages[i][1])
		cages[i][0] -= 1
		if cages[i][0] == 0:
			cages.remove(i)
	return prepared_cages


# Move to next level
func next_level() -> void:
	set_level_number(get_level_number() + 1)
	get_tree().change_scene("res://scenes/screen/Main.tscn")


# Reset level
func reset_level() -> void:
	set_level_number(get_level_number())
	get_tree().change_scene("res://scenes/screen/Main.tscn")


# Parse level configuraiton
func _parse_text_file():
	var file:File = File.new()
	# warning-ignore:return_value_discarded
	file.open("res://assets/saves/level_" + str(get_level_number()) + ".txt", File.READ)
	var data:Dictionary = parse_json(file.get_as_text())
	var x:int = 0
	var y:int = 0
	level.setup(data)
	$Track/AnimationPlayer.playback_speed = 3.0 / data.spawn_time
	instructions.set_instructions(data.instructions)
	var idx = -1
	for i in data.items:
		idx += 1
		var type = i[0]
		var count = i[1]
		var entity:Item = item.instance()
		var prepared_cages:Array = _generate_cages_from_settings(data.cages)
		simulation.prepare_cages(prepared_cages)
		entity.setup(type, count)
		selection.add_item(entity, Vector2(x, y))
		x = x + 1
		if x == 4:
			x = 0
			y = y + 1
		
		if type != "Track":
			var vial_scene = Vial.scene_from_type(type)
			var vial_node = vial_scene.instance()
			vial_node.position = vial_node.locations[idx]
			vial_node.z_index = vial_node.zindexes[idx]
			get_node("/root/Main/Vials").add_child(vial_node)
			
	# Setup maxes for vials
	for target in data.targets.min_sin_punished:
		get_node("/root/Main/Vials/" + target[0] + "Vial").target = target[1]
		
	for i in data.disabled_tiles:
		var p:Vector2 = Vector2(i[0],i[1])
		level.disabled_tiles.append(p)
	level.add_disabled_tiles()


# Signalled on exit button pressed
func _on_Exit_pressed():
	$Menu.visible = true
	pause()


# Cheat code
func _process(delta: float) -> void:
	if Input.is_action_just_pressed("next"):
		next_level()
