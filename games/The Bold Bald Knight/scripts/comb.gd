extends Node3D


func _on_area_3d_body_entered(body: Node3D) -> void:
	if body.is_in_group("player"):
		level_manager.level_4 = true
		level_manager.level_7 = true
		queue_free()
