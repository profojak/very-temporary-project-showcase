extends Node2D
class_name Simulation
# Simulation
# Simulation handles spawning and movement of cages.


var cage:PackedScene = preload("res://scenes/entities/Cage.tscn")
var spawn_delay:int = 0
var spawn_time:float = 1.0
var _counter:int = 0
var prepared_cages:Array = []
var cages:Array = []
var ready:bool = false #check for prepared cages

onready var statistics = get_node("/root/Statistics")


func _process(delta: float) -> void:
	if Input.is_action_just_pressed("win"):
		cages = []
		prepared_cages = []
	if not get_parent().get_node("Orders").is_success() and len(prepared_cages) < 6:
		get_parent().get_node("MusicPlayer").playing_warning(true)
	else:
		get_parent().get_node("MusicPlayer").playing_warning(false)
	if len(prepared_cages) < 6:
		for n in range(len(prepared_cages)+1, 7):
			if n != 0:
				get_node("/root/Main/Track/Cage" + str(n)).visible = false
	else:
		for n in range(1,7):
			get_node("/root/Main/Track/Cage" + str(n)).visible = true


# Game step, move all cages
func step() -> void:
	if _counter <=0:
		_counter = spawn_delay
		get_parent().get_parent().play_animation()
		_spawn_cage()
	_counter -= 1

	_mark_incorrect_punishment()	
	_update_cage_directions()
	_handle_cage_collisions()
	_end_marked_cages()
	_move_cages(_next_cage_positions())

	if prepared_cages.empty() and cages.empty():
		if not get_parent().get_node("LevelEndTimer").time_left:
			if get_parent().get_node("Orders").is_success():
				get_parent().get_parent().get_node("SuccessPlayer").play()
			else:
				get_parent().get_parent().get_node("FailPlayer").play()
			get_parent().get_node("LevelEndTimer").start() # Reset level or move to the next one


# If cage entered wrong torture 
func _mark_incorrect_punishment():
	var tiles:Array = get_parent().tiles
	for c in cages:
		var tile = tiles[c.grid_index.x][c.grid_index.y]
		if tile is Torture and not c.sins[tile.type]:
			c.should_end = true


# Prepare list of cages
func prepare_cages(arr:Array) -> void:
	for i in range(arr.size()):
		var prepared_cage:Cage = cage.instance()
		prepared_cage.set_sins(arr[i])
		prepared_cage.set_speed(3.0 / spawn_time)
		prepared_cages.append(prepared_cage)


# Update where cages should head next
func _update_cage_directions() -> void:
	var tiles:Array = get_parent().tiles
	for c in cages:
		var tile:Entity = tiles[c.grid_index.x][c.grid_index.y]
		if tile is Track and tile.has_junction:
			c.direction = tile.junction_dir
		elif tile is Tile:
			if tile is Torture:
				if c.sins[tile.type]:
					c.remove_sins(tile.type)
					tile.get_node("CPUParticles2D").emitting = true
			c.get_direction_from_tile(tile)
		elif c.grid_index == Vector2(-1, 0):
			# Do nothing
			pass
		#	c.direction = Entity.RIGHT
		else:
			assert(false, "Cage is on unexpected tile")


# End cells that would occupy the same cell in the next step 
# or that would collide during transfer. 
func _handle_cage_collisions() -> void:
	var positions = _current_cage_positions()
	var next_positions = _next_cage_positions()
	var to_end_transfer = _cell_collisions_transfer(positions, next_positions)
	for i in range(len(to_end_transfer) - 1, -1, -1):
		if to_end_transfer[i]:
			cages[i].should_collide = true
			cages[i].collision()
			statistics.cage_collision_count+=1
			statistics.cage_statistics[cages[i].id]["colided"]=true
			
	var to_end_occ = _cell_collisions_occupation(cages, positions, next_positions)
	for i in range(len(to_end_occ) - 1, -1, -1):
		if to_end_occ[i]:
			cages[i].should_collide = true
			cages[i].collision()
			statistics.cage_collision_count+=1
			statistics.cage_statistics[cages[i].id]["colided"]=true


# Get current cage positions
func _current_cage_positions() -> Array:
	var positions:Array = []
	for c in cages:
		positions.append(c.grid_index)
	return positions


# Get next cage positions
func _next_cage_positions() -> Array:
	var positions = []
	for c in cages:
		positions.append(c.next_postition())
	return positions


# Check for collisions on tiles currently occupied by cages
func _cell_collisions_occupation(_cages: Array, positions:Array, next_positions:Array) -> Array:
	var to_end:Array = []
	for i in len(positions):
		var should_end:bool = false
		for j in len(next_positions):
			if i == j:
				continue
			var new_pos:Vector2 = next_positions[i]
			var other_new_pos:Vector2 = next_positions[j]
			if new_pos == other_new_pos and not _cages[i].should_end and not _cages[j].should_end:
				should_end = true
		to_end.append(should_end)
	return to_end


# Check for collisions on tiles that will be occupied after next step
func _cell_collisions_transfer(positions:Array, next_positions:Array) -> Array:
	var to_end:Array = []
	for i in len(positions):
		var should_end:bool = false
		var pos:Vector2 = positions[i]
		var new_pos:Vector2 = next_positions[i]
		for j in len(positions):
			if j == i:
				continue
			var other_pos:Vector2 = positions[j]
			var other_new_pos:Vector2 = next_positions[j]
			if _does_transfer_collision_happen(pos, new_pos, other_pos, other_new_pos):
				should_end = true
		to_end.append(should_end)
	return to_end


# Check if collision happens
func _does_transfer_collision_happen(pos:Vector2, new_pos:Vector2,
		other_pos:Vector2, other_new_pos:Vector2) -> bool:
	return pos == other_new_pos and other_pos == new_pos


# Mark cages that should end by the end of step
func _end_marked_cages() -> void:
	for i in range(len(cages) - 1, -1, -1):
		var entity:Cage = cages[i]
		if entity.should_end:
			cages.erase(entity)
			entity.end()
			statistics.cage_statistics[entity.id]["end_sins"]=entity.sins
		if entity.should_collide:
			cages.erase(entity)
			statistics.cage_statistics[entity.id]["end_sins"]=entity.sins


# Move cages
func _move_cages(next_positions) -> void:
	for i in len(cages):
		cages[i].move_to(next_positions[i])
		if next_positions[i].x<0 or next_positions[i].y<0:
			cages[i].end()
			cages.remove(i)


# Spawn cage from spawn list
func _spawn_cage() -> void:
	if !prepared_cages.empty():
		var entity:Cage = prepared_cages[0]
		entity.id = statistics.count
		statistics.count += 1
		statistics.cage_statistics[entity.id]=statistics.default_cage_statistics
		statistics.cage_statistics[entity.id]["start_sins"]=entity.sins
		prepared_cages.remove(0)
		add_child(entity)
		cages.append(entity)
		entity.grid_index = Vector2(-1, 0)
		entity.global_position = get_parent().grid_index_to_global_position(entity.grid_index)
		if get_parent().tiles[0][0] == null:
			cages.erase(entity)
			entity.end()
		get_parent().get_node("MusicPlayer").cage_spawned()


# End level
func end_level():
	var orders = get_parent().get_node("Orders")
	if orders.is_success():
		get_node("/root/Main").next_level()
	else:
		get_node("/root/Main").reset_level()
