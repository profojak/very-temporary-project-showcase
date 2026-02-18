extends Entity
class_name Item
# Item in Selection menu
# Item is used to select active entity to build on Level grid.


var count:int

export var type:String # which type of entity this Item selects


# Raise stamp when selected
func raise() -> void:
	if $Stamp_up.visible:
		$Stamp_up.visible = false
		$Stamp_down.visible = true
	else:
		$Stamp_up.visible = true
		$Stamp_down.visible = false


# Setup stamp to corresponding sin
func setup(text:String, num:int) -> void:
	$Stamp_down/Sprite.texture = load("res://assets/entities/items/" + text + ".png")
	$Stamp_up/Sprite.texture = load("res://assets/entities/items/" + text + ".png")
	type = text
	count = num
	$Stamp_up/Label.text = str(count)


# Reduce count when torture is built
func reduce_count() -> bool:
	if(count > 1):
		count = count - 1
		$Stamp_up/Label.text = str(count)
		return true
	elif count == 1:
		count = count - 1
		$Stamp_up/Label.text = str(count)
		hide()
		return true
	else:
		return false


# Increase count when torture is removed
func increase_count() -> void:
		count = count + 1
		$Stamp_up/Label.text = str(count)
		if(count > 0):
			show()
