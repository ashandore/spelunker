# Imports
import numpy as np
import tensorflow as tf
from tensorflow.contrib.data import Dataset, Iterator

frames = 4

def load_tfrecord(filename):
	#Create a dataset from the file.
	return Dataset.from_tensor_slices(np.arange(1,16,1))

def pad_front(frames, dataset):
	#Create `frames-1` empty inputs and prepend them to the dataset.
	#This simulates starting from the first frame in a real game.
	pad = Dataset.from_tensor_slices(np.zeros(frames-1, dtype = np.int32))
	return pad.concatenate(dataset)

dataset = load_tfrecord("poop")
dataset = pad_front(frames, dataset)

def construct(dataset, frames, dilation):
	#We want to create a dataset that provides
	#a `frames`-sized moving window over the
	#original dataset: e.g., (1,2,3,4), then (2,3,4,5),
	#then (3,4,5,6), and so on for `frames == 4`. We do
	#this by creating `frames` datasets, each of which
	#skips the first `i` elements of the original
	#dataset. Zipping them together then produces the
	#dataset we're looking for. We can additionally
	#apply a dilation, where would see e.g. for
	#`dilation == 2` (1,2,4,8), (2,3,5,9), ...

	datasets = []
	for i in range(frames):
		datasets.append(dataset.skip(i))
	dataset = Dataset.zip(tuple(datasets))

iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()

with tf.Session() as sess:
	sess.run(iterator.initializer)
	while True:
		try:
			elem = sess.run(next_element)
			print(elem)
		except tf.errors.OutOfRangeError:
			print("End of dataset")
			break
