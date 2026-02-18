extends Node

var cage_statistics:Dictionary = {}
var count:int=0
var arrow_turns:int=0
var cage_collision_count:int=0
var default_cage_statistics:Dictionary={}
var level_number = -1


# Very nice unused piece of code
func _ready():
	default_cage_statistics["time_alive"]=0#tracks how many ticks cage is on Grid
	default_cage_statistics["rigth_factories"]=0#count of rightly processed in factories
	default_cage_statistics["false_factories"]=0#count of badly processed in factories
	default_cage_statistics["start_sins"]={}#starting sins
	default_cage_statistics["end_sins"]={}#remaining sins of cage that end()-ed XD
	default_cage_statistics["colided"]=false#true/false
	default_cage_statistics["time_alive"]=0#
