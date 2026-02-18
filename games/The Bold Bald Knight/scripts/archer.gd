extends CharacterBody3D


@onready var player = get_tree().get_first_node_in_group("player").get_node("PlayerCollider")


const arrow_scene = preload("res://scenes/arrow.tscn")


var health = 4
var dead = false
var view_range = 25
var attacking = false
var on_cooldown = false
var cooldown = 5.0


func hit() -> void:
	health -= 1
	attacking = false
	var tween = create_tween()
	$ArcherSprite.modulate = Color("red")
	tween.tween_property($ArcherSprite, "modulate", Color("white"), 0.5)
	$ArcherSprite.stop_attack()
	on_cooldown = true
	get_tree().create_timer(5.0).timeout.connect(func(): on_cooldown = false)
	if health <= 1:
		dead = $ArcherSprite.die()
		if dead:
			$ArcherCollider.set_deferred("disabled", true)
		else:
			health = 1


func shoot_arrow() -> void:
	if not attacking:
		return
	attacking = false
	$ArcherSprite.attacking = false
	on_cooldown = true
	$ArcherSprite/ArcherAnimation.play("idle")
	$ArcherSprite.texture = $ArcherSprite.idle
	get_tree().create_timer(5.0).timeout.connect(func(): on_cooldown = false)
	
	var direction = player.global_position - $ArcherCollider.global_position
	direction = direction.normalized()
	var arrow = arrow_scene.instantiate()
	get_tree().get_root().add_child(arrow)
	arrow.global_position = $ArcherCollider.global_position + direction * 2
	arrow.look_at(arrow.global_position + direction)
	arrow.launch(direction)


func draw_bow() -> void:
	attacking = true
	$ArcherSprite.draw_bow()
	get_tree().create_timer(3.0).timeout.connect(shoot_arrow)


func line_of_sight() -> bool:
	var space_state = get_world_3d().direct_space_state
	var query = PhysicsRayQueryParameters3D.create($ArcherCollider.global_position, player.global_position)
	query.exclude = [$ArcherCollider, player, player.get_parent(), player.get_parent().get_node("SwordHit")]
	var result = space_state.intersect_ray(query)
	if result.is_empty():
		return true
	else:
		return false


func _physics_process(_delta: float) -> void:
	if line_of_sight():
		var distance = $ArcherCollider.global_position.distance_to(player.global_position)
		if distance <= view_range and not attacking and on_cooldown == false:
			if not $ArcherSprite.dying and not dead:
				draw_bow()
