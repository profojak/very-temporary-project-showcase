extends Node2D
class_name Vial

export var n = 0
export var target = 20
export var type: String
# Declare member variables here. Examples:
# var a: int = 2
# var b: String = "text"

var locations: Array = [
	Vector2(26, 144),
	Vector2(58, 144),
	Vector2(90, 144),
	Vector2(122, 144),
	Vector2(26, 177),
	Vector2(58, 177),
	Vector2(90, 177),
	Vector2(122, 177)
]

var zindexes = [
	2,2,2,2,4,4,4,4
]

func _ready() -> void:
	modulate = Color("d6d6d6")
	add_child(load("res://scenes/vialhighlight.tscn").instance())


static func scene_from_type(type):
	return load("res://scenes/entities/" + type + "Vial.tscn")

var MAX_PIXEL_HEIGHT = 16
var WIDTH = 32
var old_n = 0

# Called when the node enters the scene tree for the first time.
func _process(delta: float) -> void:
	if n < target*1.2:
		$bottom.region_rect = Rect2(0, 0, 30, MAX_PIXEL_HEIGHT*n/(target*1.5))
		$bottom.position.y = -MAX_PIXEL_HEIGHT*n/(target*1.5)/2
		$up.position.y = -MAX_PIXEL_HEIGHT*n/(target*1.5) + 0
		
	
	if n >= target:
		modulate = Color("ffffff")
	else:
		if n != old_n:
			$AnimationPlayer.play("highlight")
		old_n = n
	
func highlight():
	pass
