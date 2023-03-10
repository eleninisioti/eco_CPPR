import os

import pandas as pd

from continuous_eval.gridworld_eleni import Gridworld
from jax import random
from continuous_eval.utils import VideoWriter
import jax
import jax.numpy as jnp
import numpy as np
import pickle
import matplotlib.pyplot as plt
import time
import random as nj_random

ACTION_SIZE = 5

test_configs = {"test_firstmove_low": {"grid_width": 25,
                                       "grid_length": 25,
                                       "nb_agents": 1,
                                       "hard_coded": 0,
                                       "gen_length": 800,
                                       "init_food": 10,
                                       "place_agent": True,
                                       "place_resources": True,
                                       "regrowth_scale": 0},

                "test_firstmove_medium": {"grid_width": 25,
                                          "grid_length": 25,
                                          "nb_agents": 1,
                                          "hard_coded": 0,
                                          "gen_length": 800,
                                          "init_food": 20,
                                          "place_agent": True,
                                          "place_resources": True,
                                          "regrowth_scale": 0},

                "test_firstmove_high": {"grid_width": 25,
                                        "grid_length": 25,
                                        "nb_agents": 1,
                                        "hard_coded": 0,
                                        "gen_length": 800,
                                        "init_food": 60,
                                        "place_agent": True,
                                        "place_resources": True,
                                        "regrowth_scale": 0},

                "test_foraging_low": {"grid_width": 20,
                                      "grid_length": 20,
                                      "nb_agents": 1,
                                      "hard_coded": 0,
                                      "gen_length": 800,
                                      "init_food": 10,
                                      "place_agent": False,
                                      "place_resources": False,
                                      "regrowth_scale": 0},

                "test_foraging_medium": {"grid_width": 20,
                                         "grid_length": 20,
                                         "nb_agents": 1,
                                         "hard_coded": 0,
                                         "gen_length": 300,
                                         "init_food": 30,
                                         "place_agent": False,
                                         "place_resources": False,
                                         "regrowth_scale": 0},

                "test_foraging_high": {"grid_width": 20,
                                       "grid_length": 20,
                                       "nb_agents": 1,
                                       "hard_coded": 0,
                                       "gen_length": 50,
                                       "init_food": 180,
                                       "place_agent": False,
                                       "place_resources": False,
                                       "regrowth_scale": 0},

                "test_exploration": {"grid_width": 100,
                                     "grid_length": 100,
                                     "nb_agents": 1,
                                     "hard_coded": 0,
                                     "gen_length": 300,
                                     "init_food": 250,
                                     "place_agent": True,
                                     "place_resources": True,
                                     "regrowth_scale": 0},

                }

def process_eval(total_eval_params, project_dir, current_gen):

    with open(project_dir + "/eval/data/gen_" + str(current_gen) + ".pkl", "wb") as f:
        pickle.dump(total_eval_params, f)

def process_eval_old(total_eval_params, project_dir, current_gen):
    # current_gen = len(total_eval_params)

    efficiency = {}
    sustainability = {}
    norm_efficiency = {}
    following = {}
    dispersal = {}

    for gen in range(len(total_eval_params)):
        for test_type in total_eval_params[gen].keys():
            if test_type not in efficiency.keys():
                efficiency[test_type] = [total_eval_params[gen][test_type]["efficiency"]]
            else:
                efficiency[test_type].append(total_eval_params[gen][test_type]["efficiency"])

            if test_type not in sustainability.keys():
                sustainability[test_type] = [total_eval_params[gen][test_type]["sustainability"]]
            else:
                sustainability[test_type].append(total_eval_params[gen][test_type]["sustainability"])

            if test_type not in norm_efficiency.keys():
                norm_efficiency[test_type] = [total_eval_params[gen][test_type]["norm_efficiency"]]
            else:
                norm_efficiency[test_type].append(total_eval_params[gen][test_type]["norm_efficiency"])

            """
            if test_type not in following.keys():
                following[test_type] = [total_eval_params[gen][test_type]["following"]]
            else:
                following[test_type].append(total_eval_params[gen][test_type]["following"])

            if test_type not in dispersal.keys():
                dispersal[test_type] = [total_eval_params[gen][test_type]["dispersal"]]
            else:
                dispersal[test_type].append(total_eval_params[gen][test_type]["dispersal"])
            """
        processed_results = {}
        for test_type in efficiency.keys():
            fig, axs = plt.subplots(4, figsize=(7, 12))

            axs[0].plot(range(len(efficiency[test_type])), efficiency[test_type])
            axs[0].set_ylabel("Efficiency")

            axs[0].plot(range(len(norm_efficiency[test_type])), norm_efficiency[test_type])
            axs[0].set_ylabel("Norm-Efficiency")

            axs[1].plot(range(len(sustainability[test_type])), sustainability[test_type])
            axs[1].set_ylabel("Sustainability")
            """

            axs[2].plot(range(len(following[test_type])), following[test_type])
            axs[2].set_ylabel("Following")

            axs[3].plot(range(len(dispersal[test_type])), dispersal[test_type])
            axs[3].set_ylabel("dispersal")
            """

            plt.savefig(project_dir + "/eval/" + test_type + "/media/gen_" + str(current_gen) + ".png")
            plt.clf()

    processed_results = {"efficiency": efficiency,
                         "sustainability": sustainability,
                         "norm_efficiency": norm_efficiency
                         }

    print("saving ", project_dir + "/eval/data/gen_" + str(current_gen) + ".pkl")

    with open(project_dir + "/eval/data/gen_" + str(current_gen) + ".pkl", "wb") as f:
        pickle.dump(processed_results, f)


def measure_following(agents, agent_view):
    group_following = 0
    for i, posx in enumerate(agents.posx):
        posy = agents.posy[i]
        following = 0
        for j, neighborx in enumerate(agents.posx):
            if i != j:
                dist = np.sqrt((posx - agents.posx[j]) ** 2 + (posy - agents.posy[j]) ** 2)
                if dist < np.sqrt(2 * agent_view ** 2):
                    following = 1
                    break
        group_following += following

    group_following = group_following / len(agents.posx)

    return group_following


def measure_dispersal(agents, agent_view):
    group_dispersal = 0
    for i, posx in enumerate(agents.posx):
        posy = agents.posy[i]
        distances = 0
        for j, neighborx in enumerate(agents.posx):
            distances += np.sqrt((posx - agents.posx[j]) ** 2 + (posy - agents.posy[j]) ** 2)

        group_dispersal += distances

    group_dispersal = group_dispersal / len(agents.posx)

    return group_dispersal


def eval(params, nb_train_agents, key, model, project_dir, agent_view, current_gen):
    """ Test the behavior of trained agents on specific tasks.
    """
    print("------Evaluating offline------")
    test_types = ["test_firstmove_low",
                  "test_firstmove_high",
                  "test_firstmove_medium"
                  ]
    eval_trials = 10
    random_agents = 50
    total_eval_metrics = {}
    nj_random.seed(1)
    eval_data = []
    eval_columns = ["gen", "test_type", "eval_trial", "agent_idx", "efficiency", "sustainability"]

    for test_type in test_types:

        print("Test-bed: ", test_type)
        config = test_configs[test_type]

        test_dir = project_dir + "/eval/" + test_type
        if not os.path.exists(test_dir + "/media"):
            os.makedirs(test_dir + "/media")

        test_dir = project_dir + "/eval/" + test_type
        if not os.path.exists(test_dir + "/data"):
            os.makedirs(test_dir + "/data")

        for agent_idx in range(random_agents):

            random_subset = nj_random.randrange(nb_train_agents)

            params_test = params[[random_subset], :]

            env = Gridworld(
                SX=config["grid_length"],
                SY=config["grid_width"],
                init_food=config["init_food"],
                nb_agents=config["nb_agents"],
                regrowth_scale=config["regrowth_scale"],
                place_agent=config["place_agent"],
                place_resources=config["place_resources"])

            for trial in range(eval_trials):

                next_key, key = random.split(key)
                state = env.reset(next_key)

                policy_states = model.reset(state)
                positions_log = {"posx": [],
                                 "posy": []}

                video_dir = test_dir + "/media/agent_" + str(agent_idx) + "/trial_" + str(trial)
                if not os.path.exists(video_dir):
                    os.makedirs(video_dir)

                print("check video at ", video_dir + "/gen_" + str(current_gen) + ".mp4")

                trial_dir = test_dir + "/data/agent_" + str(agent_idx) + "/trial_" + str(trial)
                if not os.path.exists(trial_dir):
                    os.makedirs(trial_dir)

                with VideoWriter(video_dir + "/gen_" + str(current_gen) + ".mp4", 5.0) as vid:

                    group_rewards = []
                    first_rewards = [None for el in range(config["nb_agents"])]
                    start = time.time()

                    for i in range(config["gen_length"]):

                        next_key, key = random.split(key)

                        actions_logit, policy_states = model.get_actions(state, params_test, policy_states)
                        actions = jax.nn.one_hot(jax.random.categorical(next_key, actions_logit), ACTION_SIZE)

                        # the first 10 agents always go right
                        for hard_agent in range(config["hard_coded"]):
                            hard_actions = jax.nn.one_hot([config["default_move"][i]], ACTION_SIZE)
                            actions = actions.at[hard_agent].set(hard_actions[0])

                        cur_state, state, reward, done = env.step(state, actions)

                        positions_log["posx"].append(state.agents.posx)
                        positions_log["posy"].append(state.agents.posy)

                        group_rewards.append(jnp.sum(reward[config["hard_coded"]:]))

                        first_times = np.where(reward > 0, i, None)

                        for idx, el in enumerate(first_times):
                            if el != None and first_rewards[idx] == None:
                                first_rewards[idx] = el

                        rgb_im = state.state[:, :, :3]
                        rgb_im = np.repeat(rgb_im, 4, axis=0)
                        rgb_im = np.repeat(rgb_im, 4, axis=1)
                        vid.add(rgb_im)

                    print(str(config["gen_length"]), " steps took ", str(time.time() - start))
                    vid.close()
                    # adding to dataframe
                    sustain = [el for el in first_rewards if el != None]
                    if not len(sustain):
                        sustain = [0]
                    eval_data.append([current_gen, test_type, trial, agent_idx, np.mean(group_rewards), np.mean(sustain)])

                    os.rename(video_dir + "/gen_" + str(current_gen) + ".mp4",
                              video_dir + "/gen_" + str(current_gen) + "_sustain_" + str(np.mean(sustain)) + ".mp4")


    eval_data = pd.DataFrame(eval_data, columns=eval_columns)

    return eval_data
