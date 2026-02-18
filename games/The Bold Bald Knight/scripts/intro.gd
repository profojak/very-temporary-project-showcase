extends CanvasLayer


func end_day():
	$Control/AnimationPlayer.play("end_day")
	$Label.text = "Day " + str(level_manager.level)
	await $Control/AnimationPlayer.animation_finished
	level_manager.next_day()
