extends CharacterBody3D

@onready var player_collider = $PlayerCollider
@onready var player_mover = $PlayerMover
@onready var player_camera = $PlayerCamera

@export var mouse_sensitivity = 0.15

const SPEED = 5.0
const JUMP_VELOCITY = 4.5
var on_cooldown = false
var health = 3


func set_cooldown() -> void:
	on_cooldown = false


func hit() -> void:
	health -= 1
	var tween = create_tween()
	$UI/Control/Hit.color = Color("ff000028")
	tween.tween_property($UI/Control/Hit, "color", Color("ff000000"), 0.8)
	if level_manager.level == 5:
		level_manager.level_5 = false


func _ready() -> void:
	Input.mouse_mode = Input.MOUSE_MODE_CAPTURED
	$UI/Control/TextureRect/Label.text = level_manager.objectives[level_manager.level]
	
	if (level_manager.level == 0):
		$UI/Control/Weapon.hide()


func _input(event: InputEvent) -> void:
	if event is InputEventMouseMotion and not level_manager.cutscene:
		rotation_degrees.y -= event.relative.x * mouse_sensitivity
		player_camera.rotation_degrees.x -= event.relative.y * mouse_sensitivity
		player_camera.rotation_degrees.x = clamp(player_camera.rotation_degrees.x, -90, 90)


func _physics_process(_delta: float) -> void:
	var input_dir = Input.get_vector("move_left", "move_right", "move_up", "move_down")
	if level_manager.cutscene:
		input_dir = Vector2.ZERO
	else:
		if Input.is_action_just_pressed("attack") and not on_cooldown:
			on_cooldown = true
			$Animation.play("sword_attack")
	var move_dir = (transform.basis * Vector3(input_dir.x, 0, input_dir.y)).normalized()
	if move_dir.is_zero_approx() and $Animation.assigned_animation == "walk":
		$Animation.pause()
	elif not on_cooldown:
		if not $Animation.is_playing():
			$Animation.play("walk")
	player_mover.set_move_dir(move_dir)


func _on_sword_hit_body_entered(body: Node3D) -> void:
	if body.is_in_group("enemy"):
		body.hit()
