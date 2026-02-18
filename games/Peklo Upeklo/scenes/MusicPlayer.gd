extends Node


var repeats = 2
var is_sec_phase = false
var screaming = false
var i = -1
var anxiety_level = -60
var building = false
var phrase = -1
var my_bar = 0
var sec_half = false
var higher = false
var cages_num = 0


# Start building phase
func _ready():
	building_phase()


# call after 50% of cages
func sec_h():
	sec_half = true


# Setup alternative phase (menu and such)
func alt_phase():
	$MixingDeskMusic2.queue_bar_transition("bridge")
	is_sec_phase = true


# Setup building phase
func building_phase():
	$MixingDeskMusic2.init_song("building_phase")
	$MixingDeskMusic2.play("building_phase")
	building = true


# Setup navigation phase
func playing_phase():
	$MixingDeskMusic2.queue_bar_transition("main")
	building = false
	cages_num = get_parent().get_node("Simulation").prepared_cages.size()


# call when cage enters a machine
func entered():
	if (!screaming):
		$MixingDeskMusic.quickplay("scream_machine")
		$MixingDeskMusic.toggle_mute("scream_machine", 0)
		$MixingDeskMusic.toggle_fade("scream_machine", 0)
		screaming = true


# call when cage lefts a machine
func left():
	if (screaming):
		$MixingDeskMusic.toggle_fade("scream_machine", 0)
		screaming = false


# call when cage drops
func dropped():
	$MixingDeskMusic.init_song("scream_drop")
	var song = $MixingDeskMusic._songname_to_int("scream_drop")
	var layer = $MixingDeskMusic._trackname_to_int(song, 0)
	$MixingDeskMusic.songs[song]._get_core().get_child(0).volume_db = -30
	$MixingDeskMusic.play("scream_drop")
	$MixingDeskMusic.fade_out_long("scream_drop", 0)
	if anxiety_level == -60:
		start_anxiety()
	increase_anxiety()


# for big mistakes
func error():
	$error.play()


# call when passing to harder lvls
func higher_lvls():
	higher = true


# call when sinner enters
func catchphrase():
	var rng = RandomNumberGenerator.new()
	rng.randomize()
	var my_random_number =randi() % 32
	match (my_random_number):
		0:
			$drop.play()
		1:
			$drop2.play()
		2:
			$drop3.play()
		3:
			$drop4.play()
		4:
			$drop5.play()
		5:
			$drop6.play()
		6:
			$drop7.play()
		7:
			$drop8.play()
		8:
			$drop9.play()
		9:
			$drop10.play()
		10:
			$drop11.play()
		11:
			$drop12.play()
		12:
			$drop13.play()
		13:
			$drop14.play()
		14:
			$drop15.play()
		15:
			$drop16.play()
		16:
			$drop17.play()
		17:
			$drop18.play()
		18:
			$drop19.play()
		19:
			$drop20.play()
		20:
			$drop21.play()
		21:
			$drop22.play()
		22:
			$drop23.play()
		23:
			$drop24.play()
		24:
			$drop25.play()
		25:
			$drop26.play()
		26:
			$drop27.play()
		27:
			$drop28.play()
		28:
			$drop29.play()
		29:
			$drop30.play()
		30:
			$drop31.play()
		31:
			$drop32.play()


# Reset anxiety level
func start_anxiety():
	anxiety_level = 0


# Increase anxiety level
func increase_anxiety():
	anxiety_level = anxiety_level + 3


# Decrease anxiety level
func release_anxiety():
	anxiety_level = -1


# Stones effect
func stones():
	$stones.play()


# Add sound effects to music when anxiety level is high
func process_high_anxiety():
	if higher:
			if anxiety_level >= 6:
				$guitar4.play()
				if anxiety_level >= 9:
					$guitar5.play()
					if anxiety_level >= 12:
						$guitar6.play()
						if anxiety_level >= 15:
							$guitar7.play()
	else:
		if anxiety_level >= 1:
			$guitar1.play()
			if anxiety_level >= 2:
				$guitar2.play()
				if anxiety_level >= 3:
					$guitar3.play()


# Signalled from various scenes
func _on_MixingDeskMusic2_bar(bar):
	#anxiety_level = anxiety_level + 1 // NAVAZAT NA PRUBEH HRY
	my_bar = my_bar + 1
	#var vol : float = (anxiety_level + 600) / 600

	#vol = 3
	#print(my_bar)
	
	if (anxiety_level >= 20):
		anxiety_level = 20
	if (anxiety_level <= -3):
		anxiety_level = -3
	var vol = anxiety_level
	#sec_h()
	#higher_lvls()
	if (!building):
		if (my_bar>(cages_num/2)):
			
			sec_h()
		if ((bar == 2) && (anxiety_level != -60)):
			release_anxiety()
		if bar == 8:
			var ran = randi() % 2
			if (ran == 1):
				$gong.play()
				stones()
			else: 
				$kostel.play()
		if (bar == 9):
			$MixingDeskMusic2._change_song("main")
		if my_bar == 18:

			alt_phase()
		if my_bar == 36:
			error()
			#$MixingDeskMusic2.toggle_mute("main", 0)
			#$beat.play() #// BREAK TO STEJNY
		if my_bar == 40:
			pass
			#$MixingDeskMusic2.toggle_mute("main", 0)
		
		if (bar % 2) == 0:
			if (higher):
				$break.play() 
				$break.volume_db = anxiety_level / 2
		else:
			process_high_anxiety()


# Function to increase lag to act like our game is heavy to run
func _process(delta):
	#TODO: Kompletně random na
	higher = get_node("/root/Statistics").level_number >= 3


# Signalled on cage spawn
func cage_spawned():
	if randf() < 0.2:
		catchphrase()


# Signalled on cage drop
func cage_dropped():
	dropped()


# Fuction
func playing_warning(b: bool):
	$error.playing = b
