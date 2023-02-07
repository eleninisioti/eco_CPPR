import os
import sys

sys.path.append(os.getcwd())


from reproduce_CPPR_continuous.gridworld import Gridworld

from reproduce_CPPR_continuous.utils import VideoWriter
import jax
import jax.numpy as jnp
from jax import random
import matplotlib.pyplot as plt
import numpy as np
import pickle
import datetime
from reproduce_CPPR_continuous.testing import eval, process_eval
from evojax.util import save_model, load_model
import yaml
from evojax.policy.base import PolicyState
from evojax.task.base import TaskState
from flax.struct import dataclass


@dataclass
class AgentStates_noparam(object):
    posx: jnp.uint16
    posy: jnp.uint16
    energy: jnp.ndarray
    time_good_level: jnp.uint16
    time_alive: jnp.uint16
    time_under_level: jnp.uint16
    alive: jnp.int8


@dataclass
class State_noparam(TaskState):
    last_actions: jnp.int8
    rewards: jnp.int8
    agents: AgentStates_noparam
    steps: jnp.int32

def state_no_params(state):
    agents=state.agents
    new_state_no_param=State_noparam(last_actions=state.last_actions, rewards=state.rewards,
          agents=AgentStates_noparam(posx=agents.posx, posy=agents.posy, energy=agents.energy, time_good_level=agents.time_good_level,
                                     time_alive=agents.time_alive, time_under_level=agents.time_under_level, alive=agents.alive),
          steps=state.steps)
    return new_state_no_param

state_no_params_fn=jax.jit(state_no_params)

def train(project_dir):
    with open(project_dir + "/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    max_steps = config["num_gens"] * config["gen_length"] + 1

    # initialize environment
    env = Gridworld(SX=config["grid_length"],
                    SY=config["grid_width"],
                    nb_agents=config["nb_agents"],
                    regrowth_scale=config["regrowth_scale"],
                    niches_scale=config["niches_scale"],
                    max_age=config["max_age"],
                    time_reproduce=config["time_reproduce"],
                    time_death=config["time_death"],
                    energy_decay=config["energy_decay"],
                    spontaneous_regrow=config["spontaneous_regrow"]
                    )
    key = jax.random.PRNGKey(config["seed"])
    next_key, key = random.split(key)
    state = env.reset(next_key)

    # initialize policy

    keep_mean_rewards = []
    keep_max_rewards = []
    #eval_params = []


    gens = list(range(config["num_gens"]))


    for gen in gens:
        if(state.agents.alive.sum()==0):
            print("All the population died")
            break

        mean_accumulated_rewards=0

        if gen % config["eval_freq"] == 0:
            vid = VideoWriter(project_dir + "/train/media/gen_" + str(gen) + ".mp4", 20.0)
            state_log = []
            #state_grid_log = []
        # start = time.time()
        for i in range(config["gen_length"]):

                state, reward = env.step(state)



                if (gen % config["eval_freq"] == 0):
                    rgb_im = state.state[:, :, :3]
                    #rgb_im = np.repeat(rgb_im, 2, axis=0)
                    #rgb_im = np.repeat(rgb_im, 2, axis=1)
                    vid.add(rgb_im)

                    mean_accumulated_rewards = mean_accumulated_rewards + (reward*state.agents.alive).sum()/(state.agents.alive.sum()+1e-10)

                    state_log.append(state_no_params_fn(state))
                    #state_grid_log.append(jnp.bool_(state.state[:,:,1]))

        # print("Training ", str(config["gen_length"]), " steps took ", str(time.time() - start))





        if gen % config["eval_freq"] == 0:

            keep_mean_rewards.append(mean_accumulated_rewards)


            vid.close()
            #with open(project_dir + "/train/data/gen_" + str(gen) + "states_grid.npy", "wb") as f:
            #    jnp.save(f,jnp.bool_(jnp.array(state_grid_log)))
            with open(project_dir + "/train/data/gen_" + str(gen) + "states.pkl", "wb") as f:
                pickle.dump({"states": state_log}, f)
            save_model(model_dir=project_dir + "/train/models", model_name="gen_" + str(gen), params=state.agents.params)

            plt.plot(range(len(keep_mean_rewards)), keep_mean_rewards, label="mean")
            plt.ylabel("Mean Training rewards")
            plt.legend()
            plt.savefig(project_dir + "/train/media/rewards_" + str(gen) + ".png")
            plt.clf()

            # run offline evaluation
            #eval_params.append(eval(state.agents.params, config["nb_agents"], key, env.model, project_dir, config["agent_view"], gen))
            #process_eval(eval_params, project_dir, gen)





if __name__ == "__main__":
    project_dir = sys.argv[1]
    if not os.path.exists(project_dir + "/train/data"):
        os.makedirs(project_dir + "/train/data")

    if not os.path.exists(project_dir + "/train/models"):
        os.makedirs(project_dir + "/train/models")

    if not os.path.exists(project_dir + "/train/media"):
        os.makedirs(project_dir + "/train/media")

    train(project_dir)
