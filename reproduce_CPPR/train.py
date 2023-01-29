import os
import sys

sys.path.append(os.getcwd())

from reproduce_CPPR.agent import MetaRnnPolicy_bcppr
from reproduce_CPPR.gridworld import Gridworld, ACTION_SIZE

from reproduce_CPPR.utils import VideoWriter
import jax
import jax.numpy as jnp
from jax import random
import matplotlib.pyplot as plt
import numpy as np
import pickle
import datetime
from reproduce_CPPR.testing import eval, process_eval
from evojax.util import save_model
import yaml
import time

def selection(params, nb_agents, key, ind_best, state):
    """ Survival of the fittest.
    """
    next_key1, next_key2, next_key3, key = random.split(key, 4)

    params = params.at[ind_best[:3 * nb_agents // 4]].set(jnp.concatenate(
        [params[ind_best[-nb_agents // 4:]] + 0.02 * jax.random.normal(next_key1, (nb_agents // 4, params.shape[1])),
         params[ind_best[-nb_agents // 4:]] + 0.02 * jax.random.normal(next_key2, (nb_agents // 4, params.shape[1])),
         params[ind_best[-nb_agents // 4:]] + 0.02 * jax.random.normal(next_key3, (nb_agents // 4, params.shape[1]))]))

    new_posx = state.agents.posx
    new_posy = state.agents.posy

    return params, new_posx, new_posy


def train(project_dir):
    with open(project_dir + "/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    max_steps = config["num_gens"] * config["gen_length"] + 1

    # initialize environment
    env = Gridworld(max_steps=max_steps,
                    SX=config["grid_length"],
                    SY=config["grid_width"],
                    init_food=config["init_food"],
                    nb_agents=config["nb_agents"],
                    regrowth_scale=config["regrowth_scale"],
                    niches_scale=config["niches_scale"])
    key = jax.random.PRNGKey(np.random.randint(42))
    next_key, key = random.split(key)
    state = env.reset(next_key)

    # initialize policy
    model = MetaRnnPolicy_bcppr(input_dim=((config["agent_view"] * 2 + 1), (config["agent_view"] * 2 + 1), 3 ),
                                hidden_dim=4,
                                output_dim=ACTION_SIZE,
                                encoder_layers=[],
                                hidden_layers=[8])

    params = jax.random.normal(
        next_key,
        (config["nb_agents"], model.num_params,),
    ) / 100

    keep_mean_rewards = []
    keep_max_rewards = []
    eval_params = []

    for gen in range(config["num_gens"]):
        next_key, key = random.split(key)
        state = env._reset_fn_pos_food(next_key,
                                       state.agents.posx,
                                       state.agents.posy,
                                       state.state[:, :, 1])

        policy_states = model.reset(state)

        accumulated_rewards = jnp.zeros(config["nb_agents"])

        if gen % config["eval_freq"] == 0:
            vid = VideoWriter(project_dir + "/train/media/gen_" + str(gen) + ".mp4", 20.0)
            state_log = []
        start = time.time()
        for i in range(config["gen_length"]):
            next_key, key = random.split(key)
            actions_logit, policy_states = model.get_actions(state, params, policy_states)
            actions = jax.nn.one_hot(jax.random.categorical(next_key, actions_logit * 50, axis=-1), ACTION_SIZE)

            _, state, reward, done = env.step(state, actions)

            accumulated_rewards = accumulated_rewards + reward

            if (gen % config["eval_freq"] == 0):

                rgb_im = state.state[:, :, :3]
                rgb_im = np.repeat(rgb_im, 2, axis=0)
                rgb_im = np.repeat(rgb_im, 2, axis=1)
                vid.add(rgb_im)

                state_log.append(state)
        print("Training ", str(config["gen_length"]), " steps took ", str(time.time() - start))

        ind_best = jnp.argsort(accumulated_rewards)

        if gen % config["eval_freq"] == 0:
            # save training data
            vid.close()
            with open(project_dir + "/train/data/gen_" + str(gen) + ".pkl", "wb") as f:
                pickle.dump(state_log, f)
            save_model(model_dir=project_dir + "/train/models", model_name="gen_" + str(gen), params=params)

            plt.plot(range(len(keep_mean_rewards)), keep_mean_rewards, label="mean")
            plt.plot(range(len(keep_max_rewards)), keep_max_rewards, label="max")
            plt.ylabel("Training rewards")
            plt.legend()
            plt.savefig(project_dir + "/train/media/rewards_" + str(gen) + ".png")
            plt.clf()

            # run offline evaluation
            eval_params.append(eval(params, ind_best, key, model, project_dir, config["agent_view"]))
            process_eval(eval_params, project_dir)

        params, posx, posy = selection(params,
                                       config["nb_agents"],
                                       key,
                                       ind_best,
                                       state)


if __name__ == "__main__":
    project_dir = sys.argv[1]

    train(project_dir)
