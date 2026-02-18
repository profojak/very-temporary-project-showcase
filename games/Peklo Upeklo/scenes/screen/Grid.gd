extends TileMap
class_name Grid
# Grid logic
# Grid contains shared logic between Level and Selection grids.


export var GRID_SIZE:Vector2

onready var OFFSET:Vector2 = self.global_position
onready var CELL_SIZE:Vector2 = self.cell_size


# Convert global position to grid index
func global_position_to_grid_index(pos:Vector2) -> Vector2:
	var local_position:Vector2 = pos - self.global_position
	var index:Vector2 = Vector2(0, 0)
	index.x = round((local_position.x - 16) / CELL_SIZE.x)
	index.y = round((local_position.y - 16) / CELL_SIZE.y)
	return index


# Convert grid index to global position
func grid_index_to_global_position(index:Vector2) -> Vector2:
	return index * CELL_SIZE + CELL_SIZE/2 + self.global_position


# Check if index is within grid
func is_in_grid(index:Vector2) -> bool:
	return index.x >= 0 and index.x < GRID_SIZE.x and index.y >= 0 and index.y < GRID_SIZE.y


# Align entity to center of indexed grid tile
func align_to_grid(entity: Entity) -> Vector2:
	var entity_grid_index:Vector2 = global_position_to_grid_index(entity.global_position)
	entity.global_position = grid_index_to_global_position(entity_grid_index)
	return entity_grid_index
