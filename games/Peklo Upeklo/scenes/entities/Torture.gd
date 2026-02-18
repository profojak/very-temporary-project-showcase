extends Tile
class_name Torture
# Torture instrument in Level grid
# Torture instrument interacts with tracks and cages.


# Setup torture
func setup(text:String) -> void:
	$Sprite.texture = load("res://assets/entities/tortures/" + text + ".png")
	$CPUParticles2D.texture = load("res://assets/entities/tortures/" + text + "Particle.png")
	type = text
