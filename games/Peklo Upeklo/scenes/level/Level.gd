extends Grid
class_name Level
# Level logic
# Level implements gameplay logic.


var controller:Controller
var controller_scene:PackedScene = load("res://scenes/entities/Controller.tscn")
var controller_active:bool = false
var controller_index:Vector2 = Vector2(-10, -10)
var tiles:Array = []
var disabled_tiles:Array = []
onready var main:Main = get_parent()
onready var simulation:Simulation = $Simulation
onready var timer:Timer = $Timer
onready var statistics = get_node("/root/Statistics")


func _ready() -> void:
	_on_ready_update_grid()


# Invoke when player clicks on grid. Tile is null when in second gameplay phase
func on_click(index:Vector2,tile:Tile,game_phase:int) -> void:
		if game_phase == 1:
			_on_click_update_tile(index, tile)
		else:
			_on_click_update_junction(index)


# Setup level data when loaded
func setup(data: Dictionary) -> void:
	simulation.spawn_delay = data.spawn_delay
	simulation.spawn_time = data.spawn_time
	$Orders.setup(data)


# Check if indexed tile is empty
func is_empty(index:Vector2) -> bool:
	return tiles[index.x][index.y] == null


# Return neighbors list sorted by up, right, down, and left direction
func get_neighbors(index:Vector2) -> Array:
	var neighbors:Array = []
	for d in [Vector2.UP, Vector2.RIGHT, Vector2.DOWN, Vector2.LEFT]:
		if index.x + d.x >= 0 and index.x + d.x < GRID_SIZE.x \
				and index.y + d.y >= 0 and index.y + d.y < GRID_SIZE.y:
			var entity:Entity = tiles[index.x + d.x][index.y + d.y]
			if entity:
				neighbors.append(entity)
			else:
				neighbors.append(null)
		else:
			neighbors.append(false)
	return neighbors


# Check if indexed tile is neighbor of tile occupied by Controller entity
func get_neighbor_of_controller(index:Vector2) -> int:
	var neighbors:Array = get_neighbors(index)
	var counter:int = 0
	for n in neighbors:
		if n is Controller:
			return counter
		counter = counter + 1
	return -1


# Switch from building to navigation page
func switch_phase() -> void:
	if controller_active:
		_remove_controller()
	$Background.hide()
	show_junctions()
	timer.wait_time = simulation.spawn_time
	timer.autostart = true
	timer.start()
	$MusicPlayer.playing_phase()


# Show junctions when navigation phase begins
func show_junctions() -> void:
	for row in tiles:
		for tile in row:
			if tile is Track:
				tile.show_junction()


# Prepare Level grid and tiles array
func _on_ready_update_grid() -> void:
	for x in GRID_SIZE.x:
		tiles.append([])
		for y in GRID_SIZE.y:
			tiles[x].append(null)
	for t in get_tree().get_nodes_in_group("Torture"):
		assert(t is Torture, "Only Torture nodes must be assigned in Torture group!")
		var torture:Torture = t
		var torture_grid_index:Vector2 = align_to_grid(torture)
		assert(is_in_grid(torture_grid_index), "Torture node is not in grid!")
		tiles[torture_grid_index.x][torture_grid_index.y] = torture


# Disable tiles on Level grid
func add_disabled_tiles() -> void:
		for i in disabled_tiles:
			var dis_tile:Sprite = Sprite.new()
			dis_tile.texture = load("res://assets/entities/tiles/Disabled.png")
			$Background.add_child(dis_tile)
			dis_tile.global_position = grid_index_to_global_position(i)


# Update tile on click
func _on_click_update_tile(index:Vector2, tile:Tile) -> void:
	if tile.enabled:
		if is_empty(index):
			if controller_active:
				var dir:int = get_neighbor_of_controller(index)
				if dir >= 0:
					controller.update_connections(dir)
				elif index == controller_index:
					main.increase_count(tile.get_type())
					_remove_controller()
					_remove_tile(index)
				else:
					_remove_controller()
			else:
				if !disabled_tiles.has(index):
					if main.reduce_count(tile.get_type()):
						_create_tile(index, tile)
		else:
			if controller_active:
				var dir:int = get_neighbor_of_controller(index)
				if dir >= 0:
					controller.update_connections(dir)
				elif index == controller_index:
					main.increase_count(tiles[index.x][index.y].tile.type)
					_remove_controller()
					_remove_tile(index)
				else:
					_remove_controller()
			else:
				if !disabled_tiles.has(index):
					_create_controller(index)


# Update junction on click
func _on_click_update_junction(pos:Vector2) -> void:
	if tiles[pos.x][pos.y] is Track:
		var track:Track = tiles[pos.x][pos.y]
		track.update_junction()
		statistics.arrow_turns+=1


# Remove Controller entity
func _remove_controller() -> void:
	tiles[controller_index.x][controller_index.y] = controller.tile
	controller_active = false
	controller_index = Vector2(-10, -10)
	remove_child(controller)


# Remove entity on indexed tile
func _remove_tile(index:Vector2) -> void:
	tiles[index.x][index.y].remove_tracks()
	remove_child(tiles[index.x][index.y])
	tiles[index.x][index.y] = null


# Create Controller entity on indexed tile
func _create_controller(index:Vector2) -> void:
	controller = controller_scene.instance()
	add_child(controller)
	controller.global_position = grid_index_to_global_position(index)
	var tile:Tile = tiles[index.x][index.y]
	tiles[index.x][index.y] = controller
	controller_active = true
	controller_index = index
	controller.setup(index, tile)


# Create Torture or Track entity on indexed tile
func _create_tile(index:Vector2, tile:Tile) -> void:
	if tile:
		tile.grid_index = index
		add_child(tile)
		tile.create_tracks()
		tile.global_position = grid_index_to_global_position(index)
		tiles[index.x][index.y] = tile


# Signalled when Timer times out
func _on_Timer_timeout() -> void:
	return
	$Simulation.step() # The order is important here, this feels better
	$Orders.step()


# Signalled from mixing desk
func _on_MixingDeskMusic2_bar(bar) -> void:
	if get_node("/root/Main").level_phase == 2:
		$Simulation.step() # The order is important here, this feels better
		$Orders.step()


# Signalled on level exit timeout
func _on_LevelEndTimer_timeout() -> void:
	$Simulation.end_level()


# Game tick to signal to mixing desk to synchronize the gameplay with music
func _process(delta: float) -> void:
	if Input.is_action_just_pressed("step"):
		_on_MixingDeskMusic2_bar(0)
