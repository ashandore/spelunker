# Imports
import numpy as np
import tensorflow as tf
import time
import model

tf.logging.set_verbosity(tf.logging.INFO)

def main(argv):
    frames = 16
    batch_size = 1
    input_shape = [320,180,3]
    model = model.build(input_shape,None,None,frames = frames)
    with tf.Session() as sess:        
        sess.run(tf.global_variables_initializer())
        iterations = 100
        input_shape = model.input_shape.as_list()[1:]
        print("Creating inputs with shape", input_shape)
        start_time = time.monotonic()
        for i in range(iterations): 
            output = model.infer(sess, {
                model.input: [np.random.rand(*input_shape)]
                })   
            print(output)
        
        end_time = time.monotonic()
        total_time = end_time - start_time
        inference_time = total_time/iterations

        print(iterations, "inferences in", total_time, "s")
        print(inference_time, "s per inference")
        print(1/inference_time, "inferences per second")
        print("up to", frames/inference_time, "frames per second without skip")
    # spelunker = tf.learn.Estimator(model_fn = model_fn, model_dir = "model")

if __name__ == "__main__":
    tf.app.run()
