extends Node



var repeats = 2
var is_sec_phase = false
var i = -1
var anxiety_level = 0


# Called when the node enters the scene tree for the first time.
func _ready():
	$MixingDeskMusic.init_song("main")

	$MixingDeskMusic.play("main")


# func thief_factory_process():

# call for second phase of song
func sec_phase():

	$MixingDeskMusic.queue_bar_transition("bridge")
	is_sec_phase = true
	
		
# call when cage enters a machine
func entered():
	
	#$MixingDeskMusic.play("spook")
	#i = (i + 1) % 5 for playing different loops everytime
	$MixingDeskMusic.init_song("scream_machine")
	$MixingDeskMusic.play("scream_machine")
	i = i + 1
	
	if i == 3:
		increase_anxiety()
	if i == 5:
		release_anxiety()


# call when cage lefts a machine
func left():
	
	$MixingDeskMusic.stop("scream_machine")

# call when cage drops
func dropped():
	$MixingDeskMusic.init_song("scream_drop")
	$MixingDeskMusic.play("scream_drop")
	
	
func increase_anxiety():

		$MixingDeskMusic.quickplay("anxiety")
		$MixingDeskMusic.bind_to_param("anxiety", 1)
		anxiety_level = anxiety_level + 100
		$MixingDeskMusic.feed_param(1, anxiety_level)
		

		
func release_anxiety():
	anxiety_level = anxiety_level - 100
	if (anxiety_level < 0):
		$MixingDeskMusic.stop("anxiety")





func _process(delta):		

	
	pass



