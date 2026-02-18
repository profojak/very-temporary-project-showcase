extends Node3D

@export var gravity = 30.0
@export var max_speed = 8.0
@export var move_accel = 1.0
@export var stop_drag = 0.08
@onready var move_drag = float(move_accel) / max_speed

@onready var character_body = get_parent()
var move_dir : Vector3


func set_move_dir(new_move_dir: Vector3) -> void:
	move_dir = new_move_dir
	move_dir.y = 0.0
	move_dir = move_dir.normalized()


func _process(delta: float) -> void:
	if character_body.velocity.y > 0.0 and character_body.is_on_ceiling():
		character_body.velocity.y = 0.0
	if not character_body.is_on_floor():
		character_body.velocity.y -= gravity * delta
	
	var drag = move_drag
	if move_dir.is_zero_approx():
		drag = stop_drag
	
	var flat_velocity = character_body.velocity
	flat_velocity.y = 0.0
	character_body.velocity += move_accel * move_dir - flat_velocity * drag
	
	character_body.move_and_slide()
