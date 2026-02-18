extends TextureButton


# Signalled on Resume button pressed
func _on_Resume_pressed():
	get_parent().visible = false
	get_parent().get_parent().resume()


# Signalled on Restart button pressed
func _on_Restart_pressed():
	get_parent().visible = false
	get_parent().get_parent().restart()


# Signalled on Main menu button pressed
func _on_MainMenu_pressed():
	# warning-ignore:return_value_discarded
	print("I am here")
	get_tree().change_scene("res://scenes/menu/MainMenu.tscn")
