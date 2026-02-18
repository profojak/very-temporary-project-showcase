extends Sprite3D

@onready var player: CharacterBody3D = get_tree().get_root().get_node("World/Player")
@onready var player_camera: Camera3D = player.get_node("PlayerCamera")


var dead_1 = preload("res://sprites/archer/archer_dead_1.png")
var dead_2 = preload("res://sprites/archer/archer_dead_2.png")
var skull = preload("res://sprites/archer/archer_skull.png")
var attack_1 = preload("res://sprites/archer/archer_attack_1.png")
var attack_2 = preload("res://sprites/archer/archer_attack_2.png")
var idle = preload("res://sprites/archer/archer_idle_1.png")


var dying = false
var fatal_blow = false
var attacking = false


func draw_bow() -> void:
	attacking = true
	$ArcherAnimation.pause()
	texture = attack_1
	await get_tree().create_timer(1.5).timeout
	if not attacking:
		return
	texture = attack_2


func die() -> bool:
	if fatal_blow:
		texture = skull
		if level_manager.level == 2:
			level_manager.level_2_update()
		return true
	dying = true
	$ArcherAnimation.pause()
	texture = dead_1
	get_tree().create_timer(1.0).timeout.connect(func():
		texture = dead_2
		fatal_blow = true
	)
	get_tree().create_timer(5.0).timeout.connect(func():
		if not get_parent().dead:
			fatal_blow = false
			dying = false
			$ArcherAnimation.play("idle")
	)
	return false


func stop_attack() -> void:
	if dying or fatal_blow:
		return
	attacking = false
	$ArcherAnimation.pause()
	texture = idle
	$ArcherAnimation.play("idle")


func _process(_delta: float) -> void:
	if dying:
		return
	var player_view_right = player_camera.global_transform.basis.x
	var to_target = (global_transform.origin - player_camera.global_transform.origin).normalized()
	if player_view_right.dot(to_target) > 0.0:
		flip_h = true
	else:
		flip_h = false
