from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf

input_shape = [320,180,3]

tf.logging.set_verbosity(tf.logging.INFO)

class Model(object):
    def __init__(self, features, shape):
        self.layers = [
            tf.reshape(features, shape)
        ]
    def add(self, kind, *args, **kwargs):
        layer = kind(self.layers[-1], *args, **kwargs)
        self.layers.append(layer)
        return layer
    @property
    def shape(self):
        return self.layers[-1].shape

def model_fn(features, labels, mode, frames = 1):
    model = Model(features["x"], [-1, frames, input_shape[0], input_shape[1], input_shape[2]])

    #Downsample each frame.
    model.add(tf.layers.conv3d,
        filters = 8,
        kernel_size = (1,5,5),
        strides = (1,2,2),
        padding = "valid",
        activation = tf.nn.relu
    )
    print(np.prod(model.shape), model.shape)

    model.add(tf.layers.conv3d,
        filters = 8,
        kernel_size = (1,5,5),
        strides = (1,2,2),
        padding = "valid",
        activation = tf.nn.relu
    )
    print(np.prod(model.shape), model.shape)

    model.add(tf.layers.conv3d,
        filters = 8,
        kernel_size = (min(2,frames),3,3),
        padding = "valid",
        activation = tf.nn.relu
    )
    frames = max(int(frames/2), 1)
    print(np.prod(model.shape), model.shape)

    model.add(tf.layers.conv3d,
        filters = 16,
        kernel_size = (min(2,frames),3,3),
        dilation_rate = (2,3,3),
        padding = "valid",
        activation = tf.nn.relu
    )
    frames = max(int(frames/2), 1)
    print(np.prod(model.shape), model.shape)

    model.add(tf.layers.conv3d,
        filters = 32,
        kernel_size = (min(2,frames),3,3),
        dilation_rate = (4,9,9),
        padding = "valid",
        activation = tf.nn.relu
    )
    frames = max(int(frames/2), 1)
    print(np.prod(model.shape), model.shape)

    model.add(tf.layers.conv3d,
        filters = 64,
        kernel_size = (min(2,frames),3,3),
        dilation_rate = (8,27,27),
        padding = "valid",
        activation = tf.nn.relu
    )
    frames = max(int(frames/2), 1)
    print(np.prod(model.shape), model.shape)

    model.add(tf.reshape, [-1, np.prod(model.shape[1:]).value])
    print(np.prod(model.shape), model.shape)

    model.add(tf.layers.dense, units = 1024, activation = tf.nn.relu)
    model.add(tf.layers.dropout, rate = 0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    model.add(tf.layers.dense, units = 1024, activation = tf.nn.relu)
    model.add(tf.layers.dropout, rate = 0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    model.add(tf.layers.dense, units = 512, activation = tf.nn.relu)
    model.add(tf.layers.dropout, rate = 0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    model.add(tf.layers.dense, units = 256, activation = tf.nn.relu)
    model.add(tf.layers.dropout, rate = 0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    model.add(tf.layers.dense, units = 64, activation = tf.nn.relu)
    model.add(tf.layers.dropout, rate = 0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    model.add(tf.layers.dense, units = 32, activation = tf.nn.relu)
    model.add(tf.layers.dropout, rate = 0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    model.add(tf.layers.dense, units = 11, activation = tf.nn.relu)
    model.add(tf.layers.dropout, rate = 0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    print(model.shape)

    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    print(total_parameters, "parameters")

    return model.layers[-1]


def main(argv):
    frames = 16
    batch_size = 1
    model = model_fn({"x":tf.cast(np.ones(batch_size*frames*np.prod(input_shape)),'float32')},None,None,frames = frames)
    with tf.Session() as sess:
        for i in range(10):    
            sess.run(tf.global_variables_initializer())
            print(sess.run(model))
    # spelunker = tf.learn.Estimator(model_fn = model_fn, model_dir = "model")

if __name__ == "__main__":
    tf.app.run()
