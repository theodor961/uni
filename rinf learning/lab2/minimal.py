import gymnasium as gym
import numpy
import random
import matplotlib.pyplot as plt
from time import sleep

env = gym.make("Taxi-v3", render_mode="human").env
env.reset()
env.render()
sleep(10)
env.close()