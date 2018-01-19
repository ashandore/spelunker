import numpy as np
import tensorflow as tf
import time
import importlib
from model import build
import harness

def main(argv):
    game_name = "sf2turbo"

    #Load the configuration python file for the game
    game_config = importlib.import_module('.'.join([game_name, 'config']))
    game = harness.Game("Snes9X v1.53 for Windows", .1, game_dir = game_name, verbosity = 1)
    
    #Wait for the game to launch.
    print("Waiting for game to launch...")
    while not game.running():
        pass

    #Perform initial game setup.
    game_config.setup(game)

    model = build(game.frame_shape(), None, None, frames = 1)
    # return
    keypresses = {}
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())     
        while True:
            state, frame = game.update(keypresses)
            if state == "start":
                frame = np.fromstring(frame.tobytes(), dtype = np.uint8)
                frame = frame.reshape(game.frame_shape())
                commands = model.infer(sess, {model.input: [[frame]]})
                commands = [1*(key > .5) for key in list(commands[0])]
                keypresses = game_config.map_keypresses(commands)
                print("playing:", keypresses)

                #TODO:
                # save each frame for backpropagation after the episode.
                # save each output for backpropagation after the episode.
                # abstract all of this stuff better.
                # handle initial game setup, inter-episode setup on a per-game basis.
                # improve mask testing.
                # do the backprop!
                # think about how best to do self-play.
                # improve model configuration & generation
                # improve handling of differently sized inputs (e.g., if we're off by a power of 2 then just auto-scale it)
                # model saving & loading
                # validate that the frame I'm giving the model is actually correct (that it isn't getting messed up by going from the
                #  raw screenshot to an image to a numpy array to a reshaped numpy array)
                # improve state transition handling (hand lambdas to the game and have it run itself? dunno.)
                # improve mapping of outputs to keypresses (stochastic? pre-determined threshold?)
                # script to convert saved frames to a video
                # lolololololol
            elif state == "lose":
                keypresses = game_config.map_keypresses([0]*10)
                print("Lost game!")
            elif state == "win":
                keypresses = game_config.map_keypresses([0]*10)
                print("Won game!")


if __name__ == "__main__":
    print("Loading tensorflow...")
    tf.app.run()
