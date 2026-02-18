extends RigidBody3D


const speed = 12


func launch(dir) -> void:
	linear_velocity = dir * speed
	get_tree().create_timer(20.0).timeout.connect(queue_free)


func _on_body_entered(body: Node) -> void:
	if body.is_in_group("player") or body.is_in_group("enemy"):
		if level_manager.level == 6 and body.is_in_group("enemy"):
			level_manager.level_6 = true
		body.hit()
	$Collider.hide()
	$Collider.set_deferred("disabled", true)
	set_deferred("contact_monitor", false)
	$Sprite3D.hide()
	$Sprite3D2.hide()
	$GPUParticles3D.emitting = false
	get_tree().create_timer(1.5).timeout.connect(queue_free)
