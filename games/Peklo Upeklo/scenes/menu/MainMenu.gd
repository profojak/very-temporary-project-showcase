extends Node2D


# Signalled on quit button pressed
func _on_Quit_pressed():
	get_tree().quit()


# Signalled on start button pressed
func _on_Start_pressed():
	# warning-ignore:return_value_discarded
	get_tree().change_scene("res://scenes/screen/Main.tscn")


# Signalled on credits button pressed
func _on_Credits_pressed() -> void:
	get_tree().change_scene("res://scenes/screen/Credits.tscn")
