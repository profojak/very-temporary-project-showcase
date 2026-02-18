extends Node3D


@onready var checkpoint_camera_target = $Checkpoint/CameraTarget
@onready var princess = $Checkpoint/Princess
@onready var princess_ribon: TextureRect = $Checkpoint/Princess/CanvasLayer/Control/Ribbon


const level_2 = preload("res://scenes/levels/2.tscn")
const level_3 = preload("res://scenes/levels/3.tscn")
const level_4 = preload("res://scenes/levels/4.tscn")
const level_5 = preload("res://scenes/levels/5.tscn")
const level_6 = preload("res://scenes/levels/6.tscn")
const level_7 = preload("res://scenes/levels/7.tscn")


func _ready() -> void:
	if level_manager.level == 2:
		var lvl2 = level_2.instantiate()
		$Levels.add_child(lvl2)
		level_manager.level_2 = [false, 0]
	elif level_manager.level == 3:
		var lvl3 = level_3.instantiate()
		$Levels.add_child(lvl3)
	elif level_manager.level == 4:
		var lvl4 = level_4.instantiate()
		$Levels.add_child(lvl4)
		level_manager.level_4 = false
	elif level_manager.level == 5:
		var lvl5 = level_5.instantiate()
		$Levels.add_child(lvl5)
		level_manager.level_5 = true
	elif level_manager.level == 6:
		var lvl6 = level_6.instantiate()
		$Levels.add_child(lvl6)
		level_manager.level_6 = true
	elif level_manager.level == 7:
		var lvl7 = level_7.instantiate()
		$Levels.add_child(lvl7)
		level_manager.level_7 = false
		$"Levels/-7".queue_free()
	$Player/PlayerCamera.current = true


func _on_checkpoint_tween() -> void:
	await princess.wake_up()
	if level_manager.level == 7:
		$Checkpoint/EndCamera.current = true
		$Checkpoint/AnimationPlayer.play("RESET")
		$Checkpoint/Princess/CanvasLayer.hide()
		return
	var tween = create_tween()
	tween.parallel().tween_property(princess_ribon, "position:y", 24, 0.4).set_trans(Tween.TRANS_QUAD)
	level_manager.level += 1
	$Intro.end_day()


func _on_checkpoint_body_entered(body: Node3D) -> void:
	if body.is_in_group("player"):
		if level_manager.level == 2 and not level_manager.level_2[0]:
			return
		elif level_manager.level == 4 and not level_manager.level_4:
			return
		elif level_manager.level == 5 and not level_manager.level_5:
			return
		elif level_manager.level == 6 and not level_manager.level_6:
			return
		body.get_node("UI/Control/Weapon").hide()
		level_manager.cutscene = true
		var camera = body.get_node("PlayerCamera")
		var tween = create_tween()
		tween.tween_property(camera, "global_position", checkpoint_camera_target.global_position, 2.0).set_trans(Tween.TRANS_QUAD)
		tween.parallel().tween_property(camera, "global_rotation", checkpoint_camera_target.global_rotation, 2.0).set_trans(Tween.TRANS_QUAD)
		tween.parallel().tween_property(princess_ribon, "position:y", -257, 2.0).set_trans(Tween.TRANS_QUAD)
		tween.tween_callback(_on_checkpoint_tween)
