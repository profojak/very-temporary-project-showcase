extends Grid
# Selection menu logic
# Selection lets player choose which entity to place on Level grid.

var selected:Item
var items:Array = []


func _ready() -> void:
	_on_ready_update_grid()


# Get type of selected Item
func on_click(index:Vector2) -> void:
	if items[index.x][index.y] and items[index.x][index.y].count > 0:
		$Sprite.texture = load("res://assets/entities/tiles/" + items[index.x][index.y].type + ".png")
		if selected:
			selected.raise()
		selected = items[index.x][index.y]
		selected.raise()
		$Name.text = selected.type


# Add item to selection box
func add_item(item:Item, index:Vector2) -> void:
	$YSort.add_child(item)
	item.global_position = grid_index_to_global_position(index)
	items[index.x][index.y] = item


# Get selected item
func get_item() -> Item:
	return selected


# Reduce item count
func reduce_selected_item_count(item_type:String) -> bool:
	for i in items:
		for j in i:
			var item:Item = j
			if item and item.type == item_type:
				if item.reduce_count():
					return true
	return false


# Incerase item count
func increase_selected_item_count(item_type:String) -> void:
	for i in items:
		for j in i:
			var item:Item = j
			if item and item.type == item_type:
				item.increase_count()


# Prepare Selection grid
func _on_ready_update_grid() -> void:
	for x in GRID_SIZE.x:
		items.append([])
		for y in GRID_SIZE.y:
			items[x].append(null)
	for i in get_tree().get_nodes_in_group("Item"):
		assert(i is Item, "Only Item nodes must be assigned in Item group!")
		var item:Item = i
		var item_grid_index:Vector2 = align_to_grid(item)
		assert(is_in_grid(item_grid_index), "Item node is not in grid!")
		items[item_grid_index.x][item_grid_index.y] = item


# Signalled on Play button pressed
func _on_Button_pressed() -> void:
	get_parent().switch_phase(2)
