extends Node2D

var _data: Dictionary
var _targets: Dictionary
var collisions = 0
var sins_punished:Dictionary
var sins_failed:Dictionary


# Setup statistics
func setup(data: Dictionary):
	_data = data
	_targets = data.targets
	sins_failed = {
		"Any": 0,
		"Lust": 0,
		"Gluttony": 0,
		"Greed": 0,
		"Sloth": 0,
		"Wrath": 0,
		"Envy": 0,
		"Pride": 0 
	}
	sins_punished = {
		"Any": 0,
		"Lust": 0,
		"Gluttony": 0,
		"Greed": 0,
		"Sloth": 0,
		"Wrath": 0,
		"Envy": 0,
		"Pride": 0 
	}


# Function called from a cage when it ends
func sin_punished(cage:Cage, sin_name: String):
	sins_punished[sin_name] += 1


# Called from a cage when it ends with unpunished sins
func sin_failed(cage:Cage, sin_name: String):
	sins_failed[sin_name] += 1


# Collisions
func cage_collided():
	collisions += 1


# Evaluate minimum sins punished before moving to next level
func evaluate_min_sin_punished() -> Array:
	var ret = []
	for entry in _targets["min_sin_punished"]:
		var sin_name = entry[0]
		var target_n = entry[1]
		ret.append(sins_punished[sin_name] >= target_n)
	return ret


# Evaluate maximum sins failed before moving to next level
func evaluate_max_sin_failed() -> Array:
	var ret = []
	for entry in _targets["max_sin_failed"]:
		var sin_name = entry[0]
		var target_n = entry[1]
		ret.append(sins_failed[sin_name] <= target_n)
	return ret


# Returns array or number based on target
func evaluate_target(category: String):
	match category:
		"min_sin_punished":
			return evaluate_min_sin_punished()
		"max_sin_failed":
			return evaluate_max_sin_failed()
		"min_collisions":
			return collisions >= _targets["min_collisions"]
		"max_collisions":
			return collisions <= _targets["max_collisions"]
		_:
			assert(false, "Target " + category + " not implemented")

# Example status
# {
#	"min_sin_punished": [
#    	["Lust", bool],
#    	["Greed", bool],
#    ]
#	"max_sin_fails": [
#		...
#   ]
#   "max_collisions": bool
#	"min_collisions": bool
# }
#
func get_status() -> Dictionary:
	var status = {}
	for key in _targets:
		status[key] = evaluate_target(key)
	return status


# If level is success
func is_success() -> bool:
	for value in get_status().values():
		if value is bool:
			if not value:
				return false
		elif value is Array:
			for b in value:
				if not b:
					return false
		else:
			assert(false, "Unexpected category value")
	return true


# Raise level of fluid in vials
func step():
	for child in get_node("/root/Main/Vials").get_children():
		if child is Vial:
			child.n = sins_punished[child.type]
