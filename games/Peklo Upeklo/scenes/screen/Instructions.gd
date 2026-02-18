extends Node2D
class_name Instructions

var opened:bool = false
var standard_position:Vector2 = position


func _ready():
	_move_aside(get_viewport_rect())


# Signalled when clicked on open button
func is_open_clicked(position:Vector2) -> bool:
	$AnimationPlayer.play("RESET")
	return _is_in_square(position, $Open.global_position, $Open.texture.get_size())


# Signalled when clicked on close button
func is_close_clicked(position:Vector2) -> bool:
	return _is_in_square(position, $Close.global_position, $Close.texture.get_size())


# Seems like something very important
func is_instructions_clicked(position:Vector2) -> bool:
	return _is_in_square(position, $Background.global_position, $Background.texture.get_size())


# Check if position is in square
func _is_in_square(clicked_position:Vector2, target_position:Vector2, target_size:Vector2) -> bool:
	var res:Vector2 = clicked_position - target_position
	if res.x < target_size.x / 2 and res.x > -target_size.x / 2 and \
			res.y < target_size.y / 2 and res.y > -target_size.y / 2:
		return true
	else:
		return false


# Set instructions when loaded
func set_instructions(instructions:String):
	$RichTextLabel.text = instructions


# Open instructions
func open() -> void:
	position = standard_position
	opened = true
	$Open.visible = false
	$Background.visible = true
	$Close.visible = true

 

func close(view:Rect2) -> void:
	_move_aside(view)
	opened = false
	$Open.visible = true
	$Background.visible = false
	$Close.visible = false


func _move_aside(view:Rect2) -> void:
	var moving_factor:Vector2 = view.size - $Open.texture.get_size()
	position.x = moving_factor.x
