extends Sprite3D


enum {
	GREET = 0,
	SURPRISE = 1,
	ANGRY = 2,
	CRY = 3,
	SHOCKED = 4,
	THINK = 5,
	BORED = 6,
	FIGHT = 7,
	FURIOUS = 8,
	MAD = 9,
	SELFISH = 10,
	TIRED = 11,
}

var faces: Array = [
	preload("res://sprites/princess/princess_greet.png"),
	preload("res://sprites/princess/princess_surprise.png"),
	preload("res://sprites/princess/princess_angry.png"),
	preload("res://sprites/princess/princess_cry.png"),
	preload("res://sprites/princess/princess_shocked.png"),
	preload("res://sprites/princess/princess_think.png"),
	preload("res://sprites/princess/princess_bored.png"),
	preload("res://sprites/princess/princess_fight.png"),
	preload("res://sprites/princess/princess_furious.png"),
	preload("res://sprites/princess/princess_mad.png"),
	preload("res://sprites/princess/princess_selfish.png"),
	preload("res://sprites/princess/princess_tired.png"),
]

var dialogue: Array = [
	[ # level 0
		["Oh, my dear prince, my savior!", GREET],
		["Oh, oh! Who... Who are you?", SURPRISE],
		["What is the meaning of this?", ANGRY],
		["You are no knight!", MAD],
		["This is all wrong!", SHOCKED],
		["Oh, what shall become of my fate?", ANGRY],
		["It seems anybody can come to save the princess these days!", FIGHT],
		["I am supposed to be rescued by a bold knight...", THINK],
		["A bold knight in a beautiful, shiny armor...", BORED],
		["A knight with a golden sword...", TIRED],
		["But instead, some peasant wanders to this tower!", FURIOUS],
		["No armor, no sword...", MAD],
		["Not even a talking donkey...", TIRED],
		["Hmm...", BORED],
	],
	[ # level 1
		["The day has finally come!", GREET],
		["Prince in a shining armor, sword in hand, has come for me!", GREET],
		["A bold knight, who fought off all the monsters!", GREET],
		["Wait!", SURPRISE],
		["Where are the heads of the fallen enemies?", ANGRY],
		["Did you just walk in?", MAD],
		["Oh, this is not supposed to happen!", THINK],
		["You shall save me from monstrosities, slay them...!", FIGHT],
		["Oh, what a nightmare...", MAD],
		["I am the princess! You must risk it all for me!", ANGRY],
		["Knights get slain by dragons in attempts to save princesses...", THINK],
		["Yet, I got no dragon to watch over me...", BORED],
		["Not even a single monster to guard me, it seems...", TIRED],
		["You shall not walk in like that, you need to fight!", MAD],
		["I am the princess, after all...", SELFISH],
		["Oh, I am everything a bold knight shall wish for...", SELFISH],
		["Oh, well...", BORED]
	],
	[ # level 2
		["The bold knight has come!", GREET],
		["He is so strong, so powerful...!", FIGHT],
		["He has slain all the monsters... all the beasts!", FIGHT],
		["Just to save me!!!", SURPRISE],
		["Oh, how long I have waited for this day to come...", TIRED],
		["All alone, scared and defenceless...", SHOCKED],
		["And the knight was so bold, so brave... so fast!", SURPRISE],
		["I was not expecting anyone to deal with those ferocious monsters...", FIGHT],
		["With such a brilliance!", GREET],
		["Oh, my dear knight, so fast, so bold...", THINK],
		["...just give me some more time!", CRY],
		["I, the princess, need some time to process...", THINK],
		["...and to prepare myself for the journey ahead of us.", SELFISH],
		["Just one day, that is all I ask you for!", GREET],
	],
	[ # level 3
		["OH!", SURPRISE],
		["Oh my, oh my... oh my dear knight!", TIRED],
		["What a pleasure it is to see my dear knight!", SHOCKED],
		["One day it took you to come, eventually...", THINK],
		["Just as I asked you, one day...", BORED],
		["...not a day more...", TIRED],
		["...", BORED],
		["How was the journey, if I may ask?", GREET],
		["I do hope no difficulties, no foes...", SURPRISE],
		["...no obstructions, no obstacles...", THINK],
		["...stood in my bold knight’s way...!", GREET],
		["I, unfortunately, have been met with difficulties only!", MAD],
		["One would not believe it, what a dire situation I am in!", CRY],
		["What a sorrow, what an agony I feel!", FURIOUS],
		["My dearest possession, the greatest gift I have received!", CRY],
		["It is gone!!!", SHOCKED],
		["My precious comb is nowhere to be found!", FURIOUS],
		["I cannot leave it behind, and I cannot be rescued...", ANGRY],
		["...with my hair all tangled up!", SELFISH],
		["One last thing I ask of my dear, bold knight...", GREET],
		["...I beg you...", CRY],
		["...find my comb, and my heart is yours!", GREET],
	],
	[ # level 4
		["MY COMB!!!", SURPRISE],
		["My beautiful... irreplaceable... comb!", CRY],
		["Thank you, thank you, you dear, bold, beau...", GREET],
		["EEWWW!!!", FURIOUS],
		["BLOOD! There is blood everywhere!", MAD],
		["I cannot... I cannot...", ANGRY],
		["This is NOT how a knight shall stand in front of a princess!", TIRED],
		["What a sight!", SHOCKED],
		["What a disgrace!", FIGHT],
		["What a SMELL!", FURIOUS],
		["My stomach...", BORED],
		["Come clean and well-groomed, as is proper!", ANGRY],
		["Just then my excellence will depart with you!", GREET],
		["You... bold, bold and... knight!", GREET],
	],
	[ # level 5
		["AH! You again...", SURPRISE],
		["I welcome you back, my bold knight...", GREET],
		["I was waiting, and waiting, and...", TIRED],
		["It took you so long to come!", BORED],
		["Aaah, I was waiting... I got so bored...", ANGRY],
		["I got so tired, fatigued even...", MAD],
		["The journey, awaiting us, must be so long...", THINK],
		["I need just a bit of amusement, just a bit of diversion!", GREET],
		["Otherwise, I might not be strong enough to journey with you...", TIRED],
		["Maybe I am not to be cherished ever again...", GREET],
		["...that is how tired and bored I got...", BORED],
		["OH! I know what will awaken me!", SURPRISE],
		["Entertain me!", GREET],
		["Make two monsters, two terrifying skeletons fight!", THINK],
		["Oh, how inventive and original I am!", GREET],
		["Make them shoot each other!", GREET],
		["What a blast this will be!!!", FIGHT],
	],
	[ # level 6
		["OH!", SURPRISE],
		["You have actually succeeded...", SHOCKED],
		["Yes, my dear bold knight, you have amused me...", GREET],
		["I WAS amused!", TIRED],
		["But the tragedy!", BORED],
		["The tragedy has struck me!", SURPRISE],
		["AGAIN!!!", FIGHT],
		["My dear knight, dear bold knight...", FIGHT],
		["...you will never believe it...", FIGHT],
		["...but my comb is lost!", FIGHT],
		["AGAIN!!!", FIGHT],
		["You must go and find it!", FIGHT],
		["Please...", FIGHT],
		["I just cannot part with it...", FIGHT],
		["My hair is so shiny when I use it...", FIGHT],
		["No other comb can compare!", FIGHT],
		["And hair, it is the most important thing!", FIGHT],
		["My hair...", FIGHT],
		["I have always hoped that my daughter would cherish,", FIGHT],
		["cherish this comb, just as much as me...", FIGHT],
	],
	[ # level 7
		["YOU HAVE FOUND IT!!!", SHOCKED],
		["My comb, yes, my comb...", SURPRISE],
		["You have once again found it...", SURPRISE],
		["I GIVE UP!!!", FIGHT],
		["I know of no other place to hide it!", ANGRY],
		["Yes, you have heard me right!!!", MAD],
		["I am sick of this, sick of you!", FURIOUS],
		["All I wanted was for a beautiful prince...", CRY],
		["...with beautiful, blond, curly hair...", BORED],
		["...beautiful, amusing and bold...", TIRED],
		["...to come and save me...", CRY],
		["And who came? You!", FIGHT],
		["The legend spoke of a bold knight...", MAD],
		["...saving the beautiful princess...", FURIOUS],
		["....NOT A BALD ONE!", FIGHT],
		["I WILL RATHER STAY HERE, HA!", FIGHT],
	]
]


signal user_action_taken

@onready var label = $CanvasLayer/Control/Label
var listen_to_input: bool = false


func _ready() -> void:
	self.hide()


func _input(event: InputEvent) -> void:
	if event is InputEventKey:
		if event.is_pressed():
			listen_to_input = false
			emit_signal("user_action_taken")
	elif event is InputEventMouseButton:
		if event.is_pressed():
			listen_to_input = false
			emit_signal("user_action_taken")


func wake_up() -> void:
	await get_tree().create_timer(1.5).timeout
	self.show()
	var current_line_index = 0
	while current_line_index < dialogue[level_manager.level].size():
		var current_line_text: String = dialogue[level_manager.level][current_line_index][0]
		texture = faces[dialogue[level_manager.level][current_line_index][1]]
		label.visible_characters = 0
		label.text = current_line_text
		var tween = create_tween()
		tween.tween_property(label, "visible_characters", current_line_text.length(),
			current_line_text.length() * 0.05).set_trans(Tween.TRANS_LINEAR)
		await tween.finished
		await self.user_action_taken
		current_line_index += 1
	if level_manager.level == 7:
		self.texture = faces[TIRED]
	else:
		self.hide()
	label.text = ""
	await get_tree().create_timer(1.0).timeout
