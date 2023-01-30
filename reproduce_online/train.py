import os
import sys

sys.path.append(os.getcwd())

import time
from reproduce_online.gridworld import Gridworld
import jax
from jax import random
from reproduce_online.utils import VideoWriter
import numpy as np

time_a = time.time()
nb_agents = 1000
env = Gridworld(SX=400, SY=200, nb_agents=nb_agents, max_age=1000)
key = jax.random.PRNGKey(np.random.randint(42))
next_key, key = random.split(key)

# reset_key=jax.random.split(next_key,nb_agents)
state = env.reset(next_key)

for j in range(10000):
    if (j % 100 == 0):
        with VideoWriter("out.mp4", 20.0) as vid:
            for i in range(1000):
                state = env.step(state)

                # print("d")
                rgb_im = state.state[:, :, :3]
                rgb_im = np.repeat(rgb_im, 2, axis=0)
                rgb_im = np.repeat(rgb_im, 2, axis=1)
                vid.add(rgb_im)

        vid.show()
    else:
        for i in range(1000):
            state = env.step(state)

print(time.time() - time_a)
