import os
import pickle
import neat
import gym
import flappy_bird_gym
import time
import numpy as np

# load the winner
with open('winner', 'rb') as f:
    c = pickle.load(f)

print('Loaded genome:')
print(c)

# Load the config file, which is assumed to live in
# the same directory as this script.
local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, 'config')
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     config_path)

net = neat.nn.FeedForwardNetwork.create(c, config)


env = flappy_bird_gym.make("FlappyBird-v0")
observation = env.reset()

done = False
while not done:
    action = np.argmax(net.activate(observation))
    observation, reward, done, info = env.step(action)
    env.render()
    time.sleep(1 / 1e6)  # FPS

print(info)