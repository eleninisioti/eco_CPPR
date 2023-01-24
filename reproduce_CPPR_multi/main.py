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


def selection(params, nb_agents, key, ind_best, state, staminas, staminas_late,
              selection_type="Gautier"):
    min_agents = nb_agents
    next_key1, next_key2, next_key3, next_key4, key = random.split(key, 5)

    if selection_type == "complex":
        posx = state.agents.posx
        posy = state.agents.posy

        next_key1, next_key2, next_key3, key = random.split(key, 4)

        # which agents will have one child
        two_offspring = [agent for agent, stamina in enumerate(staminas_late) if stamina > 0 and staminas[agent] > 0]
        nojaxrandom.shuffle(two_offspring)
        print("two offspring ", len(two_offspring))

        # which will have two
        one_offspring = [agent for agent, stamina in enumerate(staminas) if stamina > 0]
        nojaxrandom.shuffle(one_offspring)
        print("one offspring ", len(one_offspring))

        new_params = jnp.zeros((params.shape[0], params.shape[1]))
        new_posx = copy.copy(posx)
        new_posy = copy.copy(posy)
        noffsprings = 0

        for agent in two_offspring:
            if noffsprings <= params.shape[0]:

                new_params_temp = params[ind_best[-noffsprings]] + 0.02 * jax.random.normal(next_key1, (params.shape[1],))
                new_params = new_params.at[noffsprings].set(new_params_temp)
                new_posx = new_posx.at[noffsprings].set(state.agents.posx[agent])
                new_posy = new_posy.at[noffsprings].set(state.agents.posy[agent])

                noffsprings += 1
            if noffsprings <= params.shape[0]:
                new_params_temp = params[ind_best[-noffsprings]] + 0.02 * jax.random.normal(next_key2,
                                                                                            (params.shape[1],))
                new_params = new_params.at[noffsprings].set(new_params_temp)
                new_posx = new_posx.at[noffsprings].set(state.agents.posx[agent])
                new_posy = new_posy.at[noffsprings].set(state.agents.posy[agent])
                noffsprings += 1


            if noffsprings <= params.shape[0]:
                new_params_temp = params[ind_best[-noffsprings]] + 0.02 * jax.random.normal(next_key2,
                                                                                            (params.shape[1],))
                new_params = new_params.at[noffsprings].set(new_params_temp)
                new_posx = new_posx.at[noffsprings].set(state.agents.posx[agent])
                new_posy = new_posy.at[noffsprings].set(state.agents.posy[agent])
                noffsprings += 1

        for agent in one_offspring:
            if noffsprings <= params.shape[0]:
                new_params_temp = params[ind_best[-noffsprings]] + 0.02 * jax.random.normal(next_key3,
                                                                                            (params.shape[1],))
                new_params = new_params.at[noffsprings].set(new_params_temp)
                new_posx = new_posx.at[noffsprings].set(state.agents.posx[agent])
                new_posy = new_posy.at[noffsprings].set(state.agents.posy[agent])
                noffsprings += 1

                if noffsprings <= params.shape[0]:
                    new_params_temp = params[ind_best[-noffsprings]] + 0.02 * jax.random.normal(next_key3,
                                                                                                (params.shape[1],))
                    new_params = new_params.at[noffsprings].set(new_params_temp)
                    new_posx = new_posx.at[noffsprings].set(state.agents.posx[agent])
                    new_posy = new_posy.at[noffsprings].set(state.agents.posy[agent])
                    noffsprings += 1

        if noffsprings < min_agents:
            next_key1, next_key2, next_key3, key = random.split(key, 4)
            nb_agents = min_agents - noffsprings

            params = params.at[ind_best[:3*int(nb_agents // 4)]].set(jnp.concatenate(
                [params[ind_best[-int(nb_agents // 4):]] + 0.02 * jax.random.normal(next_key1,
                                                                               (int(nb_agents // 4), params.shape[1])),
                 params[ind_best[-int(nb_agents // 4):]] + 0.02 * jax.random.normal(next_key2,
                                                                               (int(nb_agents // 4), params.shape[1])),
                 params[ind_best[-int(nb_agents // 4):]] + 0.02 * jax.random.normal(next_key3,
                                                                               (int(nb_agents // 4), params.shape[1]))]))

            new_posx = state.agents.posx
            new_posy = state.agents.posy

    elif selection_type == "Gautier":
        next_key1, next_key2, next_key3, key = random.split(key, 4)

        params = params.at[ind_best[:3 * nb_agents // 4]].set(jnp.concatenate(
            [params[ind_best[-nb_agents // 4:]] + 0.02 * jax.random.normal(next_key1, (nb_agents // 4, params.shape[1])),
             params[ind_best[-nb_agents // 4:]] + 0.02 * jax.random.normal(next_key2, (nb_agents // 4, params.shape[1])),
             params[ind_best[-nb_agents // 4:]] + 0.02 * jax.random.normal(next_key3, (nb_agents // 4, params.shape[1]))]))

        new_posx = state.agents.posx
        new_posy = state.agents.posy

    return params, new_posx, new_posy

def test(params, ind_best,  key, model, project_dir, train_gen, climate_type):

    world_types = ["test_foraging", "test_exploration"]
    eval_rewards = {}
    for world_type in world_types:
        if world_type == "test_foraging":
            grid_width = 80
            grid_length = 190
            nb_test_agents = 15
            hard_coded = 10
            gen_length = 500
            init_food = 200
            rand_move = np.random.randint(4)
            place_agent = False
            place_resources = False

        elif world_type == "test_exploration":

            grid_width = 80
            grid_length = 140
            nb_test_agents = 15
            hard_coded = 0
            gen_length = 700
            init_food = 200
            rand_move = np.random.randint(4)
            place_agent = True
            place_resources = True


        eval_trials = 2
        test_dir = project_dir + "/evaluation/train_" + str(train_gen) + "_" + world_type
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)

        params_b = params[ind_best[-nb_test_agents:]]

        env = GridworldDynamic(max_steps=gen_length + 1,
                               SX=grid_width,
                               SY=grid_length,
                               init_food=init_food,
                               nb_agents=nb_test_agents,
                               climate_type=climate_type,
                               place_agent=place_agent,
                               place_resources=place_resources)

        next_key, key = random.split(key)
        state = env.reset(next_key)

        policy_states = model.reset(state)

        print("Evaluating offline")
        total_rewards = []
        for trial in range(eval_trials):

            with VideoWriter(test_dir + "/trial_" + str(trial) + ".mp4", 5.0) as vid:
                group_rewards = []

                for i in range(gen_length):
                    next_key, key = random.split(key)
                    actions_logit, policy_states = model.get_actions(state, params_b, policy_states)
                    actions = jax.nn.one_hot(jax.random.categorical(next_key, actions_logit), 4)

                    # the first 10 agents always go right
                    for hard_agent in range(hard_coded):
                        hard_actions = jax.nn.one_hot([rand_move], 4)
                        actions = actions.at[hard_agent].set(hard_actions[0])

                    cur_state, state, reward, done = env.step(state, actions)
                    group_rewards.append(jnp.sum(reward[hard_coded:]))

                    rgb_im = state.state[:, :, :3]
                    rgb_im = np.repeat(rgb_im, 20, axis=0)
                    rgb_im = np.repeat(rgb_im, 20, axis=1)
                    vid.add(rgb_im)
                vid.close()
                total_rewards.append(np.mean(group_rewards))

            print("Evaluation performance at this trial:", str(np.mean(total_rewards)))

        eval_rewards[world_type] = np.mean(total_rewards)

    return eval_rewards

def gautier():

    nb_agents = 200
    num_gens = 10000
    gen_length = 500
    grid_length = 80*2
    grid_width = 190*2
    init_food = 500
    noreset = True
    energy_discount = 0.95
    energy_thres = 0.001
    reproduce_once = 0.5*gen_length
    reproduce_two = 0.75*gen_length
    climate_type = "constant"
    selection_type = "Gautier"
    scale_niches = 1
    scale_niches_exponential = 200
    no_learning = False
    max_steps = num_gens*gen_length + 1
    env = GridworldDynamic(max_steps=max_steps, SX=grid_width, SY=grid_length, init_food=init_food,
                           nb_agents=nb_agents, climate_type=climate_type, scale_niches=scale_niches,
                           scale_niches_exponential=scale_niches_exponential)
    key = jax.random.PRNGKey(np.random.randint(42))
    next_key, key = random.split(key)

    # reset_key=jax.random.split(next_key,nb_agents)
    state = env.reset(next_key)

    if not no_learning:

        model = MetaRnnPolicy_bcppr(input_dim=((AGENT_VIEW * 2 + 1), (AGENT_VIEW * 2 + 1), 3),
                                    hidden_dim=4,
                                    output_dim=4,
                                    encoder_layers=[],
                                    hidden_layers=[8])

        params = jax.random.normal(
            next_key,
            (nb_agents, model.num_params,),
        ) / 100

        policy_states = model.reset(state)


    next_key, key = random.split(key)

    now = datetime.datetime.now()
    today = str(now.day) + "_" + str(now.month) + "_" + str(now.year)

    project_dir = "projects/" + today + "/norespawn_" + str(nb_agents) + "_climate" + climate_type + "_noreset_" + \
                  str(noreset) + "_select_" + selection_type + "_nichescale_" + str(scale_niches) + "_scaleexponential_" + str(scale_niches_exponential) + "_nolearn_" + str(no_learning)
    print(project_dir)
    if not os.path.exists(project_dir):
        os.makedirs(project_dir)
    else:
        project_dir = project_dir + "_newrun"
        if not os.path.exists(project_dir):

            os.makedirs(project_dir)


    keep_mean_rewards = []
    keep_max_rewards = []
    keep_eval_rewards = []
    keep_eval_explore_rewards = []

    keep_energy = []


    for iter_evo in range(num_gens):
        next_key, key = random.split(key)
        # reset_key=jax.random.split(next_key,params.shape[0])
        if not noreset:
            state = env.reset(next_key)
        else:
            state = env._reset_fn_pos_food(next_key, state.agents.posx, state.agents.posy, state.state[:, :, 1], iter_evo)
        if not no_learning:
            policy_states = model.reset(state)

        accumulated_rewards = jnp.zeros(nb_agents)
        accumulated_energy = jnp.ones(nb_agents)
        later_energy = jnp.ones(nb_agents)

        if (iter_evo % 10 == 0):
            with VideoWriter(project_dir + "/train_" + str(iter_evo) +".mp4", 20.0) as vid:
                for i in range(gen_length):
                    next_key, key = random.split(key)
                    if not no_learning:
                        actions_logit, policy_states = model.get_actions(state, params, policy_states)
                    else:
                        actions_logit = jax.nn.one_hot(random.randint(next_key, (nb_agents,), 0, 5), 5)

                    actions = jax.nn.one_hot(jax.random.categorical(next_key, actions_logit * 50, axis=-1), 4)

                    """"
                    for agent_row, agent in enumerate(accumulated_energy):
                        if agent == 0 or later_energy[agent_row]==0:
                            actions = actions.at[agent_row].set(np.zeros(4))
                    """

                    _, state, reward, done = env.step(state, actions)

                    accumulated_rewards = accumulated_rewards + reward
                    #print(i, accumulated_energy)

                    if i < reproduce_once:
                        accumulated_energy = accumulated_energy * energy_discount + reward
                        accumulated_energy = np.where(accumulated_energy < energy_thres, 0, accumulated_energy)
                    elif i == reproduce_once:
                        later_energy = copy.copy(accumulated_energy)
                    elif i < reproduce_two:
                        later_energy = later_energy * energy_discount + reward
                        later_energy = np.where(later_energy < energy_thres*0.8, 0, later_energy)

                    if (i % 1 == 0):
                        rgb_im = state.state[:, :, :3]
                        rgb_im = np.repeat(rgb_im, 2, axis=0)
                        rgb_im = np.repeat(rgb_im, 2, axis=1)
                        vid.add(rgb_im)

                vid.close()
        else:
            for i in range(gen_length):
                next_key, key = random.split(key)
                if not no_learning:
                    actions_logit, policy_states = model.get_actions(state, params, policy_states)
                else:
                    actions_logit = jax.nn.one_hot(random.randint(next_key, (nb_agents,), 0, 5), 5)

                actions = jax.nn.one_hot(jax.random.categorical(next_key, actions_logit * 50, axis=-1), 4)
                for agent_row, agent in enumerate(accumulated_energy):
                    if agent == 0 or later_energy[agent_row] == 0:
                        actions = actions.at[agent_row].set(np.zeros(4))

                _, state, reward, done = env.step(state, actions)

                accumulated_rewards = accumulated_rewards + reward

                if i < reproduce_once:
                    accumulated_energy = accumulated_energy * energy_discount + reward
                    accumulated_energy = np.where(accumulated_energy < energy_thres, 0, accumulated_energy)

                elif i == reproduce_once:
                    later_energy = copy.copy(accumulated_energy)
                elif i < reproduce_two:
                    later_energy = later_energy * energy_discount + reward
                    later_energy = np.where(later_energy < energy_thres * 0.8, 0, later_energy)

        keep_energy.append(np.mean(accumulated_energy))
        keep_mean_rewards.append(np.mean(accumulated_rewards))
        keep_max_rewards.append(max(accumulated_rewards))
        ind_best = jnp.argsort(accumulated_rewards)

        if (iter_evo % 10 == 0):
            print(jnp.mean(accumulated_rewards), accumulated_rewards[ind_best[-3:]], accumulated_rewards[ind_best[:3]])
            #eval_rewards = test(params, nb_agents, ind_best, grid_width, grid_length, key, model, project_dir, iter_evo, climate_type)
            if not no_learning:
                eval_rewards = test(params, ind_best,  key, model, project_dir, iter_evo, climate_type)
                keep_eval_rewards.append(eval_rewards["test_foraging"])
                keep_eval_explore_rewards.append(eval_rewards["test_exploration"])


            plt.plot(range(len(keep_mean_rewards)), keep_mean_rewards, label="mean")
            plt.plot(range(len(keep_max_rewards)), keep_max_rewards, label="max")
            plt.plot(range(len(keep_eval_rewards)), keep_eval_rewards, label="eval foraging")
            plt.plot(range(len(keep_eval_explore_rewards)), keep_eval_explore_rewards, label="eval exploration")

            plt.ylabel("rewards")
            plt.legend()
            plt.savefig(project_dir + "/rewards_" + str(iter_evo) + ".png")
            plt.clf()

        if not no_learning:

            params, posx, posy = selection(params, nb_agents, key, ind_best, state, accumulated_energy,
                                           later_energy,
                                           selection_type)



if __name__ == "__main__":
    gautier()