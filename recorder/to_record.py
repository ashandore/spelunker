
import importlib
import sys
import glob
import os
import numpy as np
import tensorflow as tf
from PIL import Image
import wave, struct

#Import game config file
if not os.path.exists(os.path.join(sys.argv[1], "config.py")):
	print("usage: {} <config_dir> <log_dir>".format(sys.argv[0]))
	sys.exit(1)
config = importlib.import_module('.'.join([sys.argv[1], 'config']))

class KeypressIterator(object):
	def __init__(self, filename):
		self.sample_rate = config.keypress_rate()
		self.filename = filename
		self.generator = self.__generator()
	def sample_period(self):
		return 1/self.sample_rate
	def __iter__(self):
		return self
	def __next__(self):
		return next(self.generator)
	def send(self, val):
		return self.generator.send(val)
	def __generator(self):
		key_states = np.zeros(len(config.inputs()))
		key_time = 0.0

		with open(self.filename, 'r') as log:
			for line in log:
				next_key_time,state,key = line.strip('\r\n').split(',')
				next_key_time = float(next_key_time)			
				key = key.lower()			
				key_idx = config.inputs().index(key)
				#Yield the current key states, the time of this keypress,
				#and the time of the next keypress
				yield key_states, key_time, next_key_time
				key_time = next_key_time
				key_states[key_idx] = 1 if state == 'p' else 0

class AudioIterator(object):
	def __init__(self, filename, width = None, sample_rate = None):
		with wave.open(filename, 'rb') as source:
			if width is None:
				width = source.getsampwidth()
			if sample_rate is None:
				sample_rate = source.getframerate()
			if source.getframerate() % sample_rate != 0:
				raise Exception("sample rate must be a divisor of the source's sample rate")
			self.stride = int(source.getframerate()/sample_rate)
		self.width = width
		self.sample_rate = sample_rate
		self.filename = filename
		self.generator = self.__generator()
	def sample_period(self):
		return 1/self.sample_rate
	def __iter__(self):
		return self
	def __next__(self):
		return next(self.generator)
	def send(self, val):
		return self.generator.send(val)
	def __generator(self):
		with wave.open(self.filename, 'rb') as source:
			source_dtype = np.dtype('u{}'.format(source.getsampwidth()))
			target_dtype = np.dtype('u{}'.format(self.width))
			source_dtype_max = np.iinfo(source_dtype).max
			target_dtype_max = np.iinfo(target_dtype).max
			source_channels = source.getnchannels()
			for idx in range(0, source.getnframes(), self.stride):
				#Get and decode the audio frames
				source_frames = np.fromstring(source.readframes(self.stride), dtype=source_dtype)[0:source_channels]
				#Resize the frames to the specified byte width
				target_frames = (target_dtype_max*(source_frames/source_dtype_max)).astype(target_dtype)
				#Yield the audio frames, the time at which they start, and the time of the next set of samples
				yield target_frames, idx/source.getframerate(), (idx+self.stride)/source.getframerate()

class FrameIterator(object):
	def __init__(self, log_dir, format = "jpg"):
		self.pattern = os.path.join(log_dir, "*.{}".format(format))
		#FIXME: for now, we only support images that are the same shape.
		#this should change.
		self.image_size = None
		self.generator = self.__generator()
	def _decode_name(self, name):
		image_name = os.path.split(name)[1]
		frame,frame_time,_ = os.path.splitext(image_name)[0].split('-')
		frame = int(frame)
		frame_time = float(frame_time)
		return frame,frame_time
	def __iter__(self):
		return self
	def __next__(self):
		return next(self.generator)
	def send(self, val):
		return self.generator.send(val)
	def __generator(self):
		name_iterator = iter(glob.glob(self.pattern))
		filename = next(name_iterator)
		while True:
			_,this_frame_time = self._decode_name(filename)
			next_filename = next(name_iterator)
			_,next_frame_time = self._decode_name(next_filename)
			
			with Image.open(filename) as image:
				width,height = image.size
				if self.image_size is None:
					self.image_size = image.size
				elif self.image_size != image.size:
					raise Exception("FrameIterator doesn't currently support logs that have frames of different sizes.")
			with open(filename, 'rb') as image:
				yield np.fromstring(image.read(), dtype = np.uint8), this_frame_time, next_frame_time
			filename = next_filename

class WindowedIterator(object):
	def __init__(self, window_time, base_iterator):
		self.iterator = base_iterator
		self.window_size = int(window_time/base_iterator.sample_period())
		self.generator = self.__generator()
	def __iter__(self):
		return self
	def __next__(self):
		return next(self.generator)
	def send(self, val):
		return self.generator.send(val)
	def __generator(self):
		#Grab a sample
		# sample, this_sample_time, next_sample_time = next(self.iterator)
		sample,this_sample_time,next_sample_time = next(self.iterator)
		#Figure out the output shape
		shape = (self.window_size, *sample.shape)
		output = np.zeros(shape, dtype = sample.dtype)

		current_time = 0.0
		target_time = 0.0

		#FIXME: I don't properly understand generators. Look it up.
		#FIXME: Next step is to decode these records and play them back (video + audio + input, graphically)
		#		as a sanity check.

		while True:
			try:
				target_time = yield output
				while current_time < target_time:
					output = np.roll(output,-1,0)
					if current_time >= next_sample_time:
						# sample,this_sample_time,next_sample_time = next(self.iterator)
						sample,this_sample_time,next_sample_time = next(self.iterator)
					output[-1,:] = sample
					current_time += self.iterator.sample_period()
			except StopIteration:
				return

class WindowedLogIterator(object):
	"""An iterator that takes a base iterator and a set of
	iterators to take windowed samples from. At each iteration,
	this iterator will yield a single sample from the base
	iterator and, for each other iterator, a window_time length
	of samples before the base iterator's sample and a window_time
	length of samples following the base iterator's sample.
	This requires each provided iterator to yield data of the form
	sample, sample_time, next_sample_time."""
	def __init__(self, base_iterator, window_time, *window_iterators):
		self.base_iter = base_iterator
		self.window_time = window_time
		self.windowed_iterators = []
		for iterator in window_iterators:
			windowed = iter(WindowedIterator(window_time, iterator))
			next(windowed)
			self.windowed_iterators.append(windowed)

		self.generator = self.__generator()
	def __iter__(self):
		return self
	def __next__(self):
		return next(self.generator)
	def send(self, val):
		return self.generator.send(val)
	def __generator(self):
		for sample, this_sample_time, next_sample_time in self.base_iter:
			outputs = []
			for iterator in self.windowed_iterators:
				outputs.append(iterator.send(this_sample_time))
				outputs.append(iterator.send(this_sample_time + self.window_time))
			value = this_sample_time, sample, *outputs
			yield value

def _bytes_feature(value):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
def _bytes_list_feature(value):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
def _float_feature(value):
	return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def main(argv):
	log_dir = argv[2]
	keypress_iter = KeypressIterator(os.path.join(log_dir, "keylog.txt"))
	audio_iter = AudioIterator(os.path.join(log_dir, "audio.wav"))
	frame_iter = FrameIterator(log_dir, "jpg")

	data_iter = WindowedLogIterator(frame_iter, 1/config.frame_rate(), keypress_iter, audio_iter)
	samples = 0
	for item in data_iter:
		t, frame, pre_keypresses, post_keypresses, pre_audio, post_audio = item
		samples += 1
		print(t, samples, frame, pre_keypresses, post_keypresses, pre_audio, post_audio)

if __name__ == "__main__":
	main(sys.argv)
	#Validate arguments
	# with wave.open(os.path.join(sys.argv[2], "sampled.wav"), 'wb') as target:
	# 	width = 2
	# 	rate = 16000
	# 	target.setnchannels(2)
	# 	target.setframerate(rate)
	# 	target.setsampwidth(width)
	# 	for sample, t, t_next in audio_iter(os.path.join(sys.argv[2], "audio.wav"), sample_rate = rate):
	# 		target.writeframes(sample.tobytes())
	# if not os.path.exists(os.path.join(sys.argv[2], "keylog.txt")):
	# 	print("usage: {} <config_dir> <log_dir>".format(sys.argv[0]))
	# 	print("ERROR: log directory does not have a keylog file.")
	# 	sys.exit(1)

	# print("Converting {} log {} to tfrecord".format(sys.argv[1], sys.argv[2]))

	# window_size = int(config.input_rate()/config.frame_rate())
	# with tf.python_io.TFRecordWriter(os.path.join(sys.argv[2], "data.tfrecord")) as writer:
	# 	for t, input_shape, pre_inputs, post_inputs, frame_shape, frame in data_iterator(sys.argv[2], window_size):
	# 		example = tf.train.Example(features = tf.train.Features(feature = {
	# 			'time': _float_feature(t),
	# 			'frame_shape': _bytes_list_feature(frame_shape),
	# 			'frame': _bytes_list_feature(frame),
	# 			'input_shape': _bytes_list_feature(input_shape),
	# 			'pre_inputs': _bytes_list_feature(pre_inputs),
	# 			'post_inputs': _bytes_list_feature(post_inputs)
	# 		}))
	# 		print("\rt={}".format(t), end='')
	# 		writer.write(example.SerializeToString())
	# print("")
