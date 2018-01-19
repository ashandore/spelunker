import numpy as np
import tensorflow as tf

class Model(object):
    def __init__(self, shape):
        self.layers = [
            tf.placeholder("float", shape)
        ]
    def add(self, kind, *args, **kwargs):
        layer = kind(self.layers[-1], *args, **kwargs)
        self.layers.append(layer)
        return layer
    @property
    def output_shape(self):
        return self.layers[-1].shape
    @property
    def input(self):
        return self.layers[0]
    @property
    def input_shape(self):
        return self.layers[0].shape
    @property
    def graph(self):
        return self.layers[-1]
    def infer(self, session, inputs):
        return session.run(self.graph, feed_dict = inputs)

def build(input_shape, labels, mode, frames = 1):
    model = Model([None, frames, input_shape[0], input_shape[1], input_shape[2]])
    #dimensions are batch size x frames per input x frame width x frame height x channels (rgb)

    
    #For each frame, let's downsample to a single-channel image of smaller size.
    #Then, we can look across frames.

    #I want a 3d convolution that looks at RGB, treating each frame as a single channel.
    #So, basically, batch size of batch_size*frames; 1 channel in and 1 channel out.

    with tf.device("/device:GPU:0"):
        model.add(tf.layers.conv3d,
            filters = 8,
            kernel_size = (1,5,5),
            strides = (1,2,2),
            padding = "valid",
            activation = tf.nn.relu
        )
        print(np.prod(model.output_shape), model.output_shape)

        model.add(tf.layers.conv3d,
            filters = 8,
            kernel_size = (1,5,5),
            strides = (1,2,2),
            padding = "valid",
            activation = tf.nn.relu
        )
        print(np.prod(model.output_shape), model.output_shape)

        model.add(tf.layers.conv3d,
            filters = 8,
            kernel_size = (1,5,5),
            strides = (1,2,2),
            padding = "valid",
            activation = tf.nn.relu
        )
        print(np.prod(model.output_shape), model.output_shape)

        model.add(tf.layers.conv3d,
            filters = 32,
            kernel_size = (min(2,frames),3,3),
            padding = "valid",
            activation = tf.nn.relu
        )
        frames = max(int(frames/2), 1)
        print(np.prod(model.output_shape), model.output_shape)

        model.add(tf.layers.conv3d,
            filters = 64,
            kernel_size = (min(2,frames),3,3),
            dilation_rate = (2,3,3),
            padding = "valid",
            activation = tf.nn.relu
        )
        frames = max(int(frames/2), 1)
        print(np.prod(model.output_shape), model.output_shape)

        model.add(tf.layers.conv3d,
            filters = 128,
            kernel_size = (min(2,frames),3,3),
            dilation_rate = (4,9,9),
            padding = "valid",
            activation = tf.nn.relu
        )
        frames = max(int(frames/2), 1)
        print(np.prod(model.output_shape), model.output_shape)

        model.add(tf.layers.conv3d,
            filters = 128,
            kernel_size = (min(2,frames),3,3),
            dilation_rate = (8,1,1),
            padding = "valid",
            activation = tf.nn.relu
        )
        frames = max(int(frames/2), 1)
        print(np.prod(model.output_shape), model.output_shape)

        model.add(tf.reshape, [-1, np.prod(model.output_shape[1:]).value])
        print(np.prod(model.output_shape), model.output_shape)

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

        model.add(tf.layers.dense, units = 10, activation = tf.nn.sigmoid)
        model.add(tf.layers.dropout, rate = 0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    print("Input shape:", model.input_shape)
    print("Output shape:", model.output_shape)

    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    print(total_parameters, "parameters")

    return model
