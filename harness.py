from PIL import Image, ImageChops, ImageStat
import os
import glob
import time
from recorder import capture
from recorder import record
from recorder import mask_test
import keyboard

def create_frame_test(window, target_path, mask_path, threshold):
	window_size = capture.get_window_size(window)
	image_size = (window_size[2], window_size[3])
	target = Image.open(target_path).convert("RGB").resize(image_size)
	mask = Image.open(mask_path).convert("RGB").resize(image_size)
	return mask_test.create_test(target, mask, threshold)

#A Game is an iterator that yields game state and a frame.
#takes keypresses as an input. If there are multiple sets of keypresses,
#it will evenly space them across the configured update period.
#The returned frame will be from the end of the update period.
#If an update rate is configured, it will do so no more often than that rate.
class Game(object):
	def __init__(self, name, period, game_dir, verbosity = 0):
		self.state = None
		# config = configparser.ConfigParser().read(config_path)

		self.window_name = name
		self.update_period = period

		#Load up frame detection images
		self.frame_tests = {}
		for filename in glob.glob(os.path.join(game_dir, "*.png")):
			filename = os.path.split(filename)[-1]
			if "_mask.png" not in filename:				
				event,ext = os.path.splitext(filename)
				target = os.path.join(game_dir, filename)
				mask = os.path.join(game_dir, event+"_mask"+ext)
				self.frame_tests[event] = create_frame_test(self.window_name, target, mask, 15)
				if verbosity > 0:
					print("loaded mask test for state", event)

		self.limiter = record.RateLimiter(self.update_period)
	def running(self):
		return capture.window_exists(self.window_name)
	def frame_shape(self):
		shape = capture.get_window_size(self.window_name)
		return shape[2], shape[3], 3
	def send_keypresses(self, inputs):
		for key, state in inputs.items():
			if state == 0 and keyboard.is_pressed(key):
				keyboard.release(key)
			elif state == 1 and not keyboard.is_pressed(key):
				keyboard.press(key)
	def update(self, inputs = None):
		#If input was provided, send it to the game
		if inputs is not None:
			self.send_keypresses(inputs)

		#Wait for the update time
		next(self.limiter)

		#Grab a frame, update state
		frame = capture.frame(self.window_name)
		for state,test in self.frame_tests.items():		
			if test(frame):
				self.state = state

		#Return the latest frame + state
		return self.state, frame


if __name__ == "__main__":
	game = Game("Snes9X v1.53 for Windows", .1, mask_dir = "mask")
	while True:
		state, frame = game.update()
