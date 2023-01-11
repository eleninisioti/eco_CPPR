import os
import sys

sys.path.append(os.getcwd())
from reproduce_CPPR.utils import VideoWriter, gini_coefficient
from reproduce_CPPR.gridworld import Gridworld
import random as nojaxrandom
import jax
import jax.numpy as jnp
from jax import random
from flax.struct import dataclass
import matplotlib.pyplot as plt
import numpy as np
from reproduce_CPPR.agent import MetaRnnPolicy_b, DensePolicy_b
import pickle
import datetime
from evojax.util import load_model
import copy
AGENT_VIEW = 3


def test(params, nb_agents, ind_best,  SX, SY, key, model, project_dir, train_gen):
    smaller_grid = True
    if smaller_grid:
        divide = 8
    else:
        divide = 4
    SX = int(640 / divide)
    SY = int(1520 / divide)
    nb_test_agents = 10
    params_b = params[ind_best[-nb_test_agents:]]
    rand_move = np.random.randint(4)
    rand_move = 3
    moves = {0: "left", 1: "down", 2: "up", 3: "right"}
    print("random move in train_gen ", str(train_gen), str(moves[rand_move]))
    init_food = 200
    env = Gridworld(100, nb_test_agents, init_food, SX, SY, climate_type="no-niches")

    next_key, key = random.split(key)
    state = env.reset(next_key)

    policy_states = model.reset(state)
    eval_trials = 2
    test_dir = project_dir + "/evaluation/train_" + str(train_gen) + "_move_" + moves[rand_move]
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    hard_coded = 5
    total_rewards = []
    for trial in range(eval_trials):
        print("trial ", str(trial))


        with VideoWriter(test_dir + "/trial_" + str(trial) + ".mp4", 4.0) as vid:
            group_rewards = []

            for i in range(750):
                next_key, key = random.split(key)
                actions_logit, policy_states = model.get_actions(state, params_b, policy_states)


                actions = jax.nn.one_hot(jax.random.categorical(next_key, actions_logit), 5)

                # my agent

                for hard_agent in range(hard_coded):
                    if i < 20:
                        hard_actions = jax.nn.one_hot([rand_move], 5)
                    else:
                        hard_actions = jax.nn.one_hot([rand_move], 5)

                    actions = actions.at[hard_agent].set(hard_actions[0])

                cur_state, state, reward, done = env.step(state, actions)
                group_rewards.append(jnp.sum(reward))

                # print(state.agents.seeds)
                rgb_im = state.state[:, :, :3]

                rgb_im = np.repeat(rgb_im, 20, axis=0)
                rgb_im = np.repeat(rgb_im, 20, axis=1)
                vid.add(rgb_im)
            vid.close()
            total_rewards.append(np.sum(group_rewards))

    print("eval performnace", str(np.mean(total_rewards)))
    return np.mean(total_rewards)

def no_training_reset():
    SX = int(640 / 8)
    SY = int(1520 / 8)
    nb_agents = 200
    num_train_gens = 1
    gen_length = 750
    init_food = 200
    climate = "no-niches"
    climate_var = 0.5
    env = Gridworld(num_train_gens * gen_length + 1, nb_agents, init_food, SX, SY, climate_type=climate,
                    climate_var=climate_var)
    key = jax.random.PRNGKey(np.random.randint(42))
    next_key, key = random.split(key)
    state = env.reset(next_key)

    plt.figure(figsize=(8, 6), dpi=160)

    vid = True
    now = datetime.datetime.now()
    today = str(now.day) + "_" + str(now.month) + "_" + str(now.year)
    project_dir = "projects/" + today + "/reset_genlen_" + str(gen_length) + "_climate_" + climate + "_var_" + str(climate_var)
    if not os.path.exists(project_dir):
        os.makedirs(project_dir)
    pos_x = state.agents.posx
    pos_y = state.agents.posy

    if (vid):
        with VideoWriter(project_dir + "/training.mp4", 20.0) as vid:
            for gen in range(num_train_gens):
                state = env._reset_fn(next_key)

                for i in range(gen_length):
                    next_key, key = random.split(key)
                    temp_actions = jax.nn.one_hot(random.randint(next_key, (nb_agents,), 0, 5), 5)
                    actions = jax.nn.one_hot([3] * nb_agents, 5)

                    cur_state, state, reward, done = env.step(state, actions)
                    # print(state.agents.seeds)

                    rgb_im = state.state[:, :, :3]
                    rgb_im = np.repeat(rgb_im, 5, axis=0)
                    rgb_im = np.repeat(rgb_im, 5, axis=1)
                    vid.add(rgb_im)
                    print("generation ", str(gen))

def no_training_small():
    SX = int(640 / 8)
    SY = int(1520 / 8)
    nb_agents = 200
    num_train_gens = 1000
    gen_length = 75
    init_food = 200
    climate = "constant"
    climate_var = 0.5
    env = Gridworld(num_train_gens * gen_length + 1, nb_agents, init_food, SX, SY, climate_type=climate,
                    climate_var=climate_var)
    key = jax.random.PRNGKey(np.random.randint(42))
    next_key, key = random.split(key)
    state = env.reset(next_key)

    plt.figure(figsize=(8, 6), dpi=160)

    vid = True
    now = datetime.datetime.now()
    today = str(now.day) + "_" + str(now.month) + "_" + str(now.year)
    project_dir = "projects/" + today + "/_genlen_" + str(gen_length) + "_climate_" + climate + "_var_" + str(climate_var)
    if not os.path.exists(project_dir):
        os.makedirs(project_dir)
    pos_x = state.agents.posx
    pos_y = state.agents.posy

    if (vid):
        with VideoWriter(project_dir + "/training.mp4", 20.0) as vid:
            for gen in range(num_train_gens):
                state = env._reset_fn_pos_food(next_key, pos_x, pos_y, state.state[:, :, 1], gen=gen)

                for i in range(gen_length):
                    next_key, key = random.split(key)
                    temp_actions = jax.nn.one_hot(random.randint(next_key, (nb_agents,), 0, 5), 5)
                    actions = jax.nn.one_hot([3] * nb_agents, 5)

                    cur_state, state, reward, done = env.step(state, actions)
                    # print(state.agents.seeds)

                rgb_im = state.state[:, :, :3]
                rgb_im = np.repeat(rgb_im, 5, axis=0)
                rgb_im = np.repeat(rgb_im, 5, axis=1)
                vid.add(rgb_im)
                print("generation ", str(gen))


def selection(params, nb_agents, key, ind_best, state, staminas, staminas_late, mutation_prob, selection_type="Gautier"):
    max_agents = nb_agents
    min_agents = nb_agents
    if selection_type == "complex":
        posx = state.agents.posx
        posy = state.agents.posy

        next_key1, next_key2, next_key3, key = random.split(key, 4)

        # which agents will have one child
        two_offspring = [agent for agent, stamina in enumerate(staminas_late) if stamina > 0 and staminas[agent]>0]
        nojaxrandom.shuffle(two_offspring)

        # which will have two
        one_offspring = [agent for agent, stamina in enumerate(staminas) if stamina > 0]
        nojaxrandom.shuffle(one_offspring)


        new_params = jnp.zeros((params.shape[0], params.shape[1]))
        new_posx = copy.copy(posx)
        new_posy = copy.copy(posy)
        noffsprings = 0
        for agent in two_offspring:
            if noffsprings <= params.shape[0]:
                temp = mutation_prob*jax.random.normal(next_key1, (new_params.shape[1],))

                new_params = new_params.at[noffsprings].set(params[agent] + temp)
                new_posx = new_posx.at[noffsprings].set(state.agents.posx[agent])
                new_posy = new_posy.at[noffsprings].set(state.agents.posy[agent])

            noffsprings += 1
            if noffsprings <= params.shape[0]:
                temp = mutation_prob * jax.random.normal(next_key1, (new_params.shape[1],))

                new_params = new_params.at[noffsprings].set(params[agent] + temp)
                new_posx = new_posx.at[noffsprings].set(state.agents.posx[agent])
                new_posy = new_posy.at[noffsprings].set(state.agents.posy[agent])
            noffsprings += 1

        for agent in one_offspring:
            if noffsprings <= params.shape[0]:
                temp = mutation_prob * jax.random.normal(next_key1, (new_params.shape[1],))

                new_params = new_params.at[noffsprings].set(params[agent] + temp)
                new_posx = new_posx.at[noffsprings].set(state.agents.posx[agent])
                new_posy = new_posy.at[noffsprings].set(state.agents.posy[agent])
                noffsprings += 1

        if noffsprings < min_agents:
            not_reproduced = [el for el in list(range(nb_agents)) if el not in two_offspring and el not in one_offspring]
            random_agents = nojaxrandom.choices(not_reproduced, k=(min_agents-noffsprings))
            for agent in random_agents:
                temp = mutation_prob * jax.random.normal(next_key1, (new_params.shape[1],))

                new_params = new_params.at[noffsprings].set(params[agent] + temp)
                new_posx = new_posx.at[noffsprings].set(state.agents.posx[agent])
                new_posy = new_posy.at[noffsprings].set(state.agents.posy[agent])







    elif selection_type == "Gautier":

        next_key1, next_key2, next_key3, key = random.split(key, 4)

        ind_orig = copy.copy(ind_best)
        params = params.at[ind_best[:3 * nb_agents // 4]].set(jnp.concatenate(
            [params[ind_best[-nb_agents // 4:]] + 0.02 * jax.random.normal(next_key1,
                                                                           (nb_agents // 4, params.shape[1])),
             params[ind_best[-nb_agents // 4:]] + 0.02 * jax.random.normal(next_key2,
                                                                           (nb_agents // 4, params.shape[1])),
             params[ind_best[-nb_agents // 4:]] + 0.02 * jax.random.normal(next_key3,
                                                                           (nb_agents // 4, params.shape[1]))]))

        ind_best = np.asarray([int(el) for idx, el in enumerate(ind_best) if staminas[idx]> 0])

        if len(ind_best) >= int(nb_agents//4):
            best_agents = ind_best[-nb_agents // 4:]
        else:
            best_agents = ind_best

        """

                                                                           

       """
        if len(best_agents):
            temp = jnp.concatenate(
                [params[best_agents] + 0.02 * jax.random.normal(next_key1, (len(best_agents), params.shape[1])),
                 params[best_agents] + 0.02 * jax.random.normal(next_key2, (len(best_agents), params.shape[1])),
                 params[best_agents] + 0.02 * jax.random.normal(next_key3, (len(best_agents), params.shape[1]))])

            params = params.at[ind_orig[:3*len(best_agents)]].set(jnp.concatenate(
                [params[best_agents] + 0.02 * jax.random.normal(next_key1, (len(best_agents), params.shape[1])),
                 params[best_agents] + 0.02 * jax.random.normal(next_key2, (len(best_agents), params.shape[1])),
                 params[best_agents] + 0.02 * jax.random.normal(next_key3, (len(best_agents), params.shape[1]))]))

    else:
        next_key1, next_key2, next_key3, key = random.split(key, 4)

        params = params.at[ind_best[:(-nb_agents // 2)]].set(params[ind_best[-nb_agents // 2:]] +
                                                             0.02 * jax.random.normal(next_key1,
                                                                                      (
                                                                                          nb_agents // 2,
                                                                                          params.shape[1])))

        params = params.at[ind_best[-nb_agents // 2:]].set(params[ind_best[-nb_agents // 2:]] +
                                                           0.02 * jax.random.normal(next_key1,
                                                                                    (nb_agents // 2, params.shape[1])))

    return params, new_posx, new_posy


def training_reset(fitness_criterion, selection_type):
    smaller_grid = True
    if smaller_grid:
        divide = 8
    else:
        divide = 4
    SX = int(640 / divide)
    SY = int(1520 / divide)
    nb_agents = 200
    num_train_gens = 200
    gen_length = 750
    init_food = 200
    policy = "metarnn"
    climate = "no-niches"
    reload = False
    smaller = False
    mutation_prob = 0.02
    env = Gridworld(num_train_gens * gen_length + 1, nb_agents, init_food, SX, SY, climate_type=climate)
    key = jax.random.PRNGKey(np.random.randint(42))
    next_key, key = random.split(key)
    state = env.reset(next_key)
    # fitness_criterion = "rewards"

    plt.figure(figsize=(8, 6), dpi=160)

    vid = True
    now = datetime.datetime.now()
    today = str(now.day) + "_" + str(now.month) + "_" + str(now.year)
    project_dir = "projects/" + today + "/staminav2_follow_" + fitness_criterion + selection_type + "_genlen_" + \
                  str(gen_length) + "_policy_" + policy + "_climate_" + climate + "_reload_" + str(
        reload) + "_smaller_" + str(smaller) + "_grid_" + str(smaller_grid) + "_mutate_" + str(mutation_prob)
    print(project_dir)
    if not os.path.exists(project_dir):
        os.makedirs(project_dir)

    input_dim = 6 + ((AGENT_VIEW * 2 + 1) ** 2) * 3
    if smaller:
        hidden_dim = 16
    else:
        hidden_dim = 128
    if policy == "metarnn":
        model = MetaRnnPolicy_b(input_dim=6 + ((AGENT_VIEW * 2 + 1) ** 2) * 3, hidden_dim=hidden_dim, output_dim=4)
        # model = MetaRnnPolicy_b(input_dim=        5 + ((AGENT_VIEW * 2 + 1) ** 2) * 3, hidden_dim=128, output_dim=4)

    else:
        model = DensePolicy_b(input_dim=6 + ((AGENT_VIEW * 2 + 1) ** 2) * 3, hidden_dim=128, output_dim=4)

    if not reload:
        params = jax.random.uniform(
            next_key,
            (nb_agents, model.num_params,),
            minval=-0.1,
            maxval=0.1,
        )
    else:

        params_single, obs_param = load_model("reproduce_CPPR/models_v1")
        params = jax.numpy.expand_dims(params_single, axis=0)
        for agent in range(nb_agents - 1):
            params = jax.numpy.append(params, jax.numpy.expand_dims(params_single, axis=0), axis=0)

    posx = state.agents.posx
    posy = state.agents.posy

    mean_rewards = []
    mean_staminas = []
    gini_coeffs = []
    eval_rewards = []
    if (vid):
        try:
            vid = VideoWriter(project_dir + "/training_0.mp4", 20.0)

            for iter in range(num_train_gens):

                next_key, key = random.split(key)
                state = env._reset_fn(next_key)

                # temp_state = env._reset_fn_pos_food(next_key, pos_x, pos_y, food)
                policy_states = model.reset(state)
                accumulated_rewards = jnp.zeros(params.shape[0])
                accumulated_staminas = jnp.ones(params.shape[0])

                accumulated_staminas_late = jnp.ones(params.shape[0])


                for trial in range(gen_length):
                    next_key, key = random.split(key)
                    actions_logit, policy_states = model.get_actions(state, params, policy_states)
                    actions = jax.nn.one_hot(jax.random.categorical(next_key, actions_logit, axis=-1), 5)
                    none_actions = np.zeros((nb_agents, 5))
                    for agent_row, agent in enumerate(accumulated_staminas):
                        if agent == 0:
                            actions = actions.at[agent_row].set(np.zeros(5))
                    # actions = np.where(accumulated_staminas == 0, actions, actions)
                    cur_state, state, reward, done = env.step(state, actions)
                    accumulated_rewards = accumulated_rewards + reward

                    if trial < int(0.7*gen_length):
                        accumulated_staminas = accumulated_staminas * 0.92 + reward
                        # print(sum(reward), sum(accumulated_staminas))
                        accumulated_staminas = np.where(accumulated_staminas < 0.01, 0, accumulated_staminas)
                    elif trial == int(0.7*gen_length):
                        accumulated_staminas_late = accumulated_staminas
                    elif trial > int(0.7*gen_length):
                        accumulated_staminas_late = accumulated_staminas_late * 0.92 + reward
                        accumulated_staminas_late = np.where(accumulated_staminas_late < 0.01, 0, accumulated_staminas_late)

                    rgb_im = state.state[:, :, :3]
                    rgb_im = np.repeat(rgb_im, 5, axis=0)
                    rgb_im = np.repeat(rgb_im, 5, axis=1)
                    vid.add(rgb_im)

                # accumulated_staminas = jnp.ones(nb_agents)
                # accumulated_rewards = jnp.ones(nb_agents)
                gini_coeffs.append(gini_coefficient(accumulated_rewards))

                mean_rewards.append(jnp.mean(accumulated_rewards))
                mean_staminas.append(jnp.mean(accumulated_staminas))
                if fitness_criterion == "stamina":
                    accumulated_rewards = accumulated_staminas

                ind_best = jnp.argsort(accumulated_rewards)

                if (iter % 1 == 0):
                    print("generation ", str(iter), str(mean_rewards[-1]))
                    print(jnp.mean(accumulated_rewards), accumulated_rewards[ind_best[-50:]],
                          accumulated_rewards[ind_best[:50]])
                print("selecting")
                params, posx, posy = selection(params, nb_agents, key, ind_best, state,accumulated_staminas, accumulated_staminas_late, mutation_prob,
                                   selection_type=selection_type)
                print("selected")
                """
                posx = state.agents.posx
                posy = state.agents.posy

                posx = jnp.concatenate(
                    [state.agents.posx[ind_best[-nb_agents // 4:]], state.agents.posx[ind_best[-nb_agents // 4:]],
                     state.agents.posx[ind_best[-nb_agents // 4:]], state.agents.posx[ind_best[-nb_agents // 4:]]])
                posy = jnp.concatenate(
                    [state.agents.posy[ind_best[-nb_agents // 4:]], state.agents.posy[ind_best[-nb_agents // 4:]],
                     state.agents.posy[ind_best[-nb_agents // 4:]], state.agents.posy[ind_best[-nb_agents // 4:]]])
                """

                rgb_im = state.state[:, :, :3]
                rgb_im = np.repeat(rgb_im, 5, axis=0)
                rgb_im = np.repeat(rgb_im, 5, axis=1)
                vid.add(rgb_im)

                if (iter % 1 == 0):
                    vid.close()
                    vid = VideoWriter(project_dir + "/training_" + str(iter) + ".mp4", 20.0)
                    with open(project_dir + "/data_" + str(iter) + ".pkl", "wb") as f:
                        pickle.dump([mean_rewards, mean_staminas, gini_coeffs], file=f)

                    plt.plot(range(len(mean_rewards)), mean_rewards)
                    plt.ylabel("reward")
                    plt.savefig(project_dir + "/rewards_" + str(iter) + ".png")
                    plt.clf()

                    plt.plot(range(len(mean_staminas)), mean_staminas)
                    plt.ylabel("reward")
                    plt.savefig(project_dir + "/staminas_" + str(iter) + ".png")
                    plt.clf()

                    plt.plot(range(len(gini_coeffs)), gini_coeffs)
                    plt.ylabel("Equalilty")
                    plt.savefig(project_dir + "/equality_" + str(iter) + ".png")
                    plt.clf()

                if (iter % 5 == 0):
                    eval_rewards.append(test(params, nb_agents, ind_best, SX, SY, key, model, project_dir, iter))


        except KeyboardInterrupt:
            print("running aborted")
            vid.close()

        with open(project_dir + "/data.pkl", "wb") as f:
            pickle.dump([mean_rewards, mean_staminas, gini_coeffs, eval_rewards], file=f)

        with open(project_dir + "/for_eval.pkl", "wb") as f:
            pickle.dump([params, nb_agents, ind_best, SX, SY, key, project_dir, iter], file=f)

        test(params, nb_agents, ind_best, SX, SY, key, model, project_dir, iter)

        plt.plot(range(len(eval_rewards)), eval_rewards)
        plt.ylabel("eval rewards")
        plt.savefig(project_dir + "/eval_rewards.png")
        plt.clf()

        plt.plot(range(len(mean_rewards)), mean_rewards)
        plt.ylabel("reward")
        plt.savefig(project_dir + "/rewards.png")
        plt.clf()

        plt.plot(range(len(mean_staminas)), mean_staminas)
        plt.ylabel("stamina")
        plt.savefig(project_dir + "/staminas.png")
        plt.clf()

        plt.plot(range(len(gini_coeffs)), gini_coeffs)
        plt.ylabel("Equalilty")
        plt.savefig(project_dir + "/equality.png")
        plt.clf()



def stable_training(fitness_criterion, selection_type):
    smaller_grid = True
    if smaller_grid:
        divide = 8
    else:
        divide = 4
    SX = int(640 / divide)
    SY = int(1520 / divide)
    nb_agents = 200
    num_train_gens = 2000
    gen_length = 75
    init_food = 200
    policy = "metarnn"
    climate = "constant"
    reload = False
    smaller = False
    env = Gridworld(num_train_gens * gen_length + 1, nb_agents, init_food, SX, SY, climate_type=climate)
    key = jax.random.PRNGKey(np.random.randint(42))
    next_key, key = random.split(key)
    state = env.reset(next_key)
    # fitness_criterion = "rewards"

    plt.figure(figsize=(8, 6), dpi=160)

    vid = True
    now = datetime.datetime.now()
    today = str(now.day) + "_" + str(now.month) + "_" + str(now.year)
    project_dir = "projects/" + today + "/follow_" + fitness_criterion + selection_type + "_genlen_" + \
                  str(gen_length) + "_policy_" + policy + "_climate_" + climate + "_reload_" + str(reload) + "_smaller_" + str(smaller) + "_grid_" + str(smaller_grid)
    print(project_dir)
    if not os.path.exists(project_dir):
        os.makedirs(project_dir)

    input_dim = 6 + ((AGENT_VIEW * 2 + 1) ** 2) * 3
    if smaller:
        hidden_dim = 16
    else:
        hidden_dim = 128
    if policy == "metarnn":
        model = MetaRnnPolicy_b(input_dim=6 + ((AGENT_VIEW * 2 + 1) ** 2) * 3, hidden_dim=hidden_dim, output_dim=4)
        #model = MetaRnnPolicy_b(input_dim=        5 + ((AGENT_VIEW * 2 + 1) ** 2) * 3, hidden_dim=128, output_dim=4)

    else:
        model = DensePolicy_b(input_dim=6 + ((AGENT_VIEW * 2 + 1) ** 2) * 3, hidden_dim=128, output_dim=4)

    if not reload:
        params = jax.random.uniform(
            next_key,
            (nb_agents, model.num_params,),
            minval=-0.1,
            maxval=0.1,
        )
    else:

        params_single, obs_param = load_model("reproduce_CPPR/models_v1")
        params =  jax.numpy.expand_dims(params_single, axis=0)
        for agent in range(nb_agents-1):
            params = jax.numpy.append(params, jax.numpy.expand_dims(params_single, axis=0), axis=0)

    posx = state.agents.posx
    posy = state.agents.posy

    mean_rewards = []
    mean_staminas = []
    gini_coeffs = []
    eval_rewards = []
    if (vid):
        try:
            vid = VideoWriter(project_dir + "/training_0.mp4", 20.0)

            for iter in range(num_train_gens):

                next_key, key = random.split(key)
                state = env._reset_fn_pos_food(next_key, posx, posy, state.state[:, :, 1], gen=iter)

                # temp_state = env._reset_fn_pos_food(next_key, pos_x, pos_y, food)
                policy_states = model.reset(state)
                accumulated_rewards = jnp.zeros(params.shape[0])
                accumulated_staminas = jnp.ones(params.shape[0])

                for trial in range(gen_length):
                    next_key, key = random.split(key)
                    actions_logit, policy_states = model.get_actions(state, params, policy_states)
                    actions = jax.nn.one_hot(jax.random.categorical(next_key, actions_logit, axis=-1), 5)
                    none_actions = np.zeros((nb_agents, 5))
                    for agent_row, agent in enumerate(accumulated_staminas):
                        if agent == 0:
                            actions = actions.at[agent_row].set(np.zeros(5))
                    #actions = np.where(accumulated_staminas == 0, actions, actions)
                    cur_state, state, reward, done = env.step(state, actions)
                    accumulated_rewards = accumulated_rewards + reward
                    accumulated_staminas = accumulated_staminas * 0.9 + reward
                    #print(sum(reward), sum(accumulated_staminas))
                    accumulated_staminas = np.where(accumulated_staminas < 0.01, 0, accumulated_staminas)

                    rgb_im = state.state[:, :, :3]
                    rgb_im = np.repeat(rgb_im, 5, axis=0)
                    rgb_im = np.repeat(rgb_im, 5, axis=1)
                    vid.add(rgb_im)

                #accumulated_staminas = jnp.ones(nb_agents)
                #accumulated_rewards = jnp.ones(nb_agents)
                gini_coeffs.append(gini_coefficient(accumulated_rewards))

                mean_rewards.append(jnp.mean(accumulated_rewards))
                mean_staminas.append(jnp.mean(accumulated_staminas))
                if fitness_criterion == "stamina":
                    accumulated_rewards = accumulated_staminas

                ind_best = jnp.argsort(accumulated_rewards)

                if (iter % 10 == 0):
                    print("generation ", str(iter), str(mean_rewards[-1]))
                    print(jnp.mean(accumulated_rewards), accumulated_rewards[ind_best[-50:]],
                          accumulated_rewards[ind_best[:50]])

                params = selection(params, nb_agents, key, ind_best, accumulated_staminas, selection_type=selection_type)
                posx = state.agents.posx
                posy = state.agents.posy

                posx = jnp.concatenate(
                    [state.agents.posx[ind_best[-nb_agents // 4:]], state.agents.posx[ind_best[-nb_agents // 4:]],
                     state.agents.posx[ind_best[-nb_agents // 4:]], state.agents.posx[ind_best[-nb_agents // 4:]]])
                posy = jnp.concatenate(
                    [state.agents.posy[ind_best[-nb_agents // 4:]], state.agents.posy[ind_best[-nb_agents // 4:]],
                     state.agents.posy[ind_best[-nb_agents // 4:]], state.agents.posy[ind_best[-nb_agents // 4:]]])

                rgb_im = state.state[:, :, :3]
                rgb_im = np.repeat(rgb_im, 5, axis=0)
                rgb_im = np.repeat(rgb_im, 5, axis=1)
                vid.add(rgb_im)

                if (iter % 10 == 0):
                    vid.close()
                    vid = VideoWriter(project_dir + "/training_" + str(iter) + ".mp4", 20.0)
                    with open(project_dir + "/data_" + str(iter) + ".pkl", "wb") as f:
                        pickle.dump([mean_rewards, mean_staminas, gini_coeffs], file=f)

                    plt.plot(range(len(mean_rewards)), mean_rewards)
                    plt.ylabel("reward")
                    plt.savefig(project_dir + "/rewards_" + str(iter) + ".png")
                    plt.clf()

                    plt.plot(range(len(mean_staminas)), mean_staminas)
                    plt.ylabel("reward")
                    plt.savefig(project_dir + "/staminas_" + str(iter) + ".png")
                    plt.clf()

                    plt.plot(range(len(gini_coeffs)), gini_coeffs)
                    plt.ylabel("Equalilty")
                    plt.savefig(project_dir + "/equality_" + str(iter) + ".png")
                    plt.clf()

                if (iter % 50 == 0):

                    eval_rewards.append(test(params, nb_agents, ind_best, SX, SY, key, model, project_dir, iter))


        except KeyboardInterrupt:
            print("running aborted")
            vid.close()

        with open(project_dir + "/data.pkl", "wb") as f:
            pickle.dump([mean_rewards, mean_staminas, gini_coeffs, eval_rewards], file=f)

        with open(project_dir + "/for_eval.pkl", "wb") as f:
            pickle.dump([params, nb_agents, ind_best, SX, SY, key, project_dir, iter], file=f)

        test(params, nb_agents, ind_best, SX, SY, key, model, project_dir, iter)


        plt.plot(range(len(eval_rewards)), eval_rewards)
        plt.ylabel("eval rewards")
        plt.savefig(project_dir + "/eval_rewards.png")
        plt.clf()

        plt.plot(range(len(mean_rewards)), mean_rewards)
        plt.ylabel("reward")
        plt.savefig(project_dir + "/rewards.png")
        plt.clf()

        plt.plot(range(len(mean_staminas)), mean_staminas)
        plt.ylabel("stamina")
        plt.savefig(project_dir + "/staminas.png")
        plt.clf()

        plt.plot(range(len(gini_coeffs)), gini_coeffs)
        plt.ylabel("Equalilty")
        plt.savefig(project_dir + "/equality.png")
        plt.clf()


if __name__ == "__main__":
    # stable_no_training()
    #stable_no_training_small()
    #no_training_reset()
    rewards = "reward"
    selection_type = "complex"
    # rewards = "stamina"
    # selection_type = "gautier"
    print(selection_type, rewards)
    training_reset(rewards, selection_type)

    # stable_training("staminas", "eleni")
    # static_world(100)
