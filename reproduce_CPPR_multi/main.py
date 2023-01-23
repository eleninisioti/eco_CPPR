import os
import sys

sys.path.append(os.getcwd())

from reproduce_CPPR_multi.agent import MetaRnnPolicy_bcppr
#from reproduce_CPPR_multi.gridworld import Gridworld
from reproduce_CPPR_multi.gridworld_dynamic import GridworldDynamic

from reproduce_CPPR_multi.utils import VideoWriter
import random as nojaxrandom
import jax
import jax.numpy as jnp
from jax import random
from flax.struct import dataclass
import matplotlib.pyplot as plt
import numpy as np
import pickle
import datetime
from evojax.util import load_model
import copy

AGENT_VIEW = 3


def test(params, nb_agents, ind_best,  SX, SY, key, model, project_dir, train_gen, climate_type):
    smaller_grid = True
    if smaller_grid:
        divide = 8
    else:
        divide = 4
    grid_width = int(640 / divide)
    grid_length = int(1520 / divide)
    nb_test_agents = 15
    params_b = params[ind_best[-nb_test_agents:]]
    rand_move = np.random.randint(4)
    max_steps = 30
    rand_move = 3
    moves = {0: "left", 1: "down", 2: "up", 3: "right"}
    print("random move in train_gen ", str(train_gen), str(moves[rand_move]))
    init_food = 200


    env = GridworldDynamic(max_steps=max_steps, SX=grid_width, SY=grid_length, init_food=init_food,
                           nb_agents=nb_test_agents, climate_type=climate_type)

    next_key, key = random.split(key)
    state = env.reset(next_key)

    policy_states = model.reset(state)
    eval_trials = 2
    test_dir = project_dir + "/evaluation/train_" + str(train_gen) + "_move_" + moves[rand_move]
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    hard_coded = 10
    total_rewards = []
    for trial in range(eval_trials):
        print("trial ", str(trial))


        with VideoWriter(test_dir + "/trial_" + str(trial) + ".mp4", 5.0) as vid:
            group_rewards = []

            for i in range(750):
                next_key, key = random.split(key)
                actions_logit, policy_states = model.get_actions(state, params_b, policy_states)


                actions = jax.nn.one_hot(jax.random.categorical(next_key, actions_logit), 4)

                # my agent

                for hard_agent in range(hard_coded):
                    if i < 20:
                        hard_actions = jax.nn.one_hot([rand_move], 4)
                    else:
                        hard_actions = jax.nn.one_hot([rand_move], 4)

                    actions = actions.at[hard_agent].set(hard_actions[0])

                cur_state, state, reward, done = env.step(state, actions)
                group_rewards.append(jnp.sum(reward[hard_coded:]))

                # print(state.agents.seeds)
                rgb_im = state.state[:, :, :3]

                rgb_im = np.repeat(rgb_im, 20, axis=0)
                rgb_im = np.repeat(rgb_im, 20, axis=1)
                vid.add(rgb_im)
            vid.close()
            total_rewards.append(np.sum(group_rewards))

    print("eval performnace", str(np.mean(total_rewards)))
    return np.mean(total_rewards)

def gautier():

    nb_agents = 200
    num_gens = 10000
    gen_length = 500
    grid_length = 80*2
    grid_width = 190*2
    init_food = 400
    noreset = False
    climate_type = "constant"
    max_steps = num_gens*gen_length + 1
    env = GridworldDynamic(max_steps=max_steps, SX=grid_width, SY=grid_length, init_food=init_food,
                           nb_agents=nb_agents, climate_type=climate_type)
    key = jax.random.PRNGKey(np.random.randint(42))
    next_key, key = random.split(key)

    # reset_key=jax.random.split(next_key,nb_agents)
    state = env.reset(next_key)

    model = MetaRnnPolicy_bcppr(input_dim=((AGENT_VIEW * 2 + 1), (AGENT_VIEW * 2 + 1), 3), hidden_dim=4, output_dim=4,
                                encoder_layers=[], hidden_layers=[8])

    next_key, key = random.split(key)

    now = datetime.datetime.now()
    today = str(now.day) + "_" + str(now.month) + "_" + str(now.year)

    project_dir = "projects/" + today + "/multi_agent_dynamic_" + str(nb_agents) + "_noreset_climate" + climate_type + "_noreset_" + str(noreset)
    print(project_dir)
    if not os.path.exists(project_dir):
        os.makedirs(project_dir)

    params = jax.random.normal(
        next_key,
        (nb_agents, model.num_params,),
    ) / 100

    policy_states = model.reset(state)
    keep_mean_rewards = []
    keep_max_rewards = []
    keep_eval_rewards = []


    for iter_evo in range(num_gens):
        next_key, key = random.split(key)
        # reset_key=jax.random.split(next_key,params.shape[0])
        if not noreset:
            state = env.reset(next_key)
        else:
            state = env._reset_fn_pos_food(next_key, state.agents.posx, state.agents.posy, state.state[:, :, 1], iter_evo)
        policy_states = model.reset(state)
        accumulated_rewards = jnp.zeros(params.shape[0])
        if (iter_evo % 10 == 0):
            with VideoWriter(project_dir + "/train_" + str(iter_evo) +".mp4", 20.0) as vid:
                for i in range(gen_length):
                    next_key, key = random.split(key)
                    actions_logit, policy_states = model.get_actions(state, params, policy_states)
                    actions = jax.nn.one_hot(jax.random.categorical(next_key, actions_logit * 50, axis=-1), 4)
                    _, state, reward, done = env.step(state, actions)

                    accumulated_rewards = accumulated_rewards + reward
                    if (i % 1 == 0):
                        rgb_im = state.state[:, :, :3]
                        rgb_im = np.repeat(rgb_im, 2, axis=0)
                        rgb_im = np.repeat(rgb_im, 2, axis=1)
                        vid.add(rgb_im)

                vid.show()
        else:
            for i in range(gen_length):
                next_key, key = random.split(key)
                actions_logit, policy_states = model.get_actions(state, params, policy_states)
                actions = jax.nn.one_hot(jax.random.categorical(next_key, actions_logit * 50, axis=-1), 4)
                _, state, reward, done = env.step(state, actions)

                accumulated_rewards = accumulated_rewards + reward

        keep_mean_rewards.append(np.mean(accumulated_rewards))
        keep_max_rewards.append(max(accumulated_rewards))
        ind_best = jnp.argsort(accumulated_rewards)
        if (iter_evo % 10 == 0):
            print(jnp.mean(accumulated_rewards), accumulated_rewards[ind_best[-3:]], accumulated_rewards[ind_best[:3]])



            eval_rewards = test(params, nb_agents, ind_best, grid_width, grid_length, key, model, project_dir, iter_evo, climate_type)

            keep_eval_rewards.append(eval_rewards)

            plt.plot(range(len(keep_mean_rewards)), keep_mean_rewards, label="mean")
            plt.plot(range(len(keep_max_rewards)), keep_max_rewards, label="max")
            plt.plot(range(len(keep_eval_rewards)), keep_eval_rewards, label="eval")


            plt.ylabel("rewards")
            plt.savefig(project_dir + "/rewards_" + str(iter_evo) + ".png")
            plt.clf()

        next_key1, next_key2, next_key3, key = random.split(key, 4)
        params = params.at[ind_best[:3 * nb_agents // 4]].set(jnp.concatenate(
            [params[ind_best[-nb_agents // 4:]] + 0.02 * jax.random.normal(next_key1, (nb_agents // 4, params.shape[1])),
             params[ind_best[-nb_agents // 4:]] + 0.02 * jax.random.normal(next_key2, (nb_agents // 4, params.shape[1])),
             params[ind_best[-nb_agents // 4:]] + 0.02 * jax.random.normal(next_key3, (nb_agents // 4, params.shape[1]))]))


if __name__ == "__main__":
    gautier()